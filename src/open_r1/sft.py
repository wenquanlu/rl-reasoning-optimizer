# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys
import torch
import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers import GenerationConfig

from open_r1.configs import SFTConfig
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

from trl.models import unwrap_model_for_generation
from trl.data_utils import apply_chat_template
from transformers.trainer import EvalLoopOutput
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers.utils import is_datasets_available
from trl.trainer.utils import pad
from transformers.data.data_collator import DataCollatorMixin
from dataclasses import dataclass
import re

# def extract_hash_answer(text: str) -> str | None:
#     if "####" not in text:
#         return None
#     return text.split("####")[1].strip()

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_hash_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return None

@dataclass
class DataCollatorForInference(DataCollatorMixin):
    pad_token_id: int
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples):
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]

        # Pad
        output = {}
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["attention_mask"] = pad(
            attention_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        )

        if "solution" in examples[0]:
            output["solution"] = [example["solution"] for example in examples]

        return output

class SFTLightEvalTrainer(SFTTrainer):
    
    # copied from Trainer.py with _remove_unused_columns commented
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        pad_token = self.args.pad_token or self.processing_class.pad_token or self.processing_class.eos_token
        pad_token_id = self.processing_class.convert_tokens_to_ids(pad_token)
        data_collator = DataCollatorForInference(pad_token_id, self.args.pad_to_multiple_of)
        
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            #eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
            pass
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def evaluation_loop(
        self,
        dataloader=None,
        description: str = "Evaluation",
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        self.model.eval()
        eval_generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=self.processing_class.pad_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            #cache_implementation=args.cache_implementation,
        )
        total = 0
        correct = 0
        preds = []
        labels = []
        with torch.no_grad():
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator
            ) as unwrapped_model:
                
                for batch in tqdm(dataloader, desc=description):
                    prompt_ids = batch["input_ids"]
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=batch["attention_mask"], generation_config=eval_generation_config
                    )
                    #print(prompt_completion_ids, "prompt completion ids!")

                    # Compute prompt length and extract completion ids
                    prompt_length = prompt_ids.size(1)
                    completion_ids = prompt_completion_ids[:, prompt_length:]
                    completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                    #print(batch)
                    targets = batch.get("solution", None)

                    for pred, target in zip(completions, targets):
                        extracted_pred = extract_hash_answer(pred)
                        if extracted_pred is not None:
                            preds.append(extracted_pred.strip())
                            labels.append(target.strip())
                            if extracted_pred.strip() == target.strip():
                                correct += 1
                        total += 1
                    # print(preds)
                    # print(labels)
                    # print(correct)
        accuracy = correct / total if total > 0 else 0.0
        metrics = {f"{metric_key_prefix}_accuracy": accuracy}
        return EvalLoopOutput(
            predictions=preds,
            label_ids=labels,
            metrics=metrics,
            num_samples=total,
        )
            

logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load datasets
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    #def rename(example):
    #    return {"prompt": example["question"], "completion": example["answer"]}
    #dataset = dataset.map(rename,
    #            remove_columns=["question", "answer"])
    def format_chat_gsm8k(example):
        return {
            "messages": [
                {"role": "user", "content": example["question"].strip()},
                {"role": "assistant", "content": example["answer"].strip()},
            ]
        }

    def make_conversation(example, prompt_column="question"):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        solution = extract_hash_answer(example["answer"])
        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt, "solution": solution}

    train_dataset = dataset[script_args.dataset_train_split].map(format_chat_gsm8k,
                remove_columns=["question", "answer"])
    eval_dataset = dataset[script_args.dataset_test_split].map(make_conversation, 
                remove_columns=["question", "answer"])
    #print(next(iter(eval_dataset['test'])), "#!!!!!")
    
    

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.padding_side = 'left'
    print("tokenizer padding side:", tokenizer.padding_side )

    eval_dataset = eval_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer}
    )

    #print(next(iter(eval_dataset['test'])), "222222")


    def tokenize(example, processing_class):
        prompts_text = example["prompt"]
        prompt_inputs = processing_class(
            text=prompts_text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False
        )
        #prompt_inputs = super()._prepare_inputs(prompt_inputs)
        #prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_inputs['input_ids'] = prompt_inputs['input_ids'][:, -512 :].squeeze(0)
        prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'][:, -512 :].squeeze(0)
        return prompt_inputs

    eval_dataset = eval_dataset.map(
        tokenize,
        fn_kwargs={
            "processing_class": tokenizer
        },
        remove_columns=["prompt"]
    )

    print(next(iter(eval_dataset)), "!333333")

    ###################
    # Load model
    ###################
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, using ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTLightEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=(eval_dataset if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
