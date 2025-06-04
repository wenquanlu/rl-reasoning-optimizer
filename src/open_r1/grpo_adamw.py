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

import logging
import os
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
import torch

logger = logging.getLogger(__name__)

from transformers import TrainerCallback
from transformers.data.data_collator import DataCollatorMixin
from trl.models import unwrap_model_for_generation
from trl.data_utils import apply_chat_template
from transformers.trainer import EvalLoopOutput
from tqdm import tqdm
from trl.trainer.utils import pad
import wandb
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers.utils import is_datasets_available
from dataclasses import dataclass
from open_r1.utils.math_eval import remove_boxed, last_boxed_only_string
import re
from transformers import GenerationConfig

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_hash_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return None

def extract_boxed_answer(text):
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    answer = remove_boxed(boxed)
    answer = answer.replace(",", "").strip()
    return answer

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

class GRPOEvalTrainer(GRPOTrainer):
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
        pad_token = self.processing_class.pad_token or self.processing_class.eos_token
        pad_token_id = self.processing_class.convert_tokens_to_ids(pad_token)
        data_collator = DataCollatorForInference(pad_token_id, None)

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
            #dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
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
                        extracted_pred = extract_boxed_answer(pred)
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
            
class GradientMonitorCallback(TrainerCallback):
    def __init__(self):
        self.grad_running_mean = None
        self.grad_running_mean_squared = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        accelerator = kwargs["accelerator"]

        # this is step before current step, because step increments after this callback in trainer
        step = state.global_step

        # Collect all gradients in a single flattened vector
        grads = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return  # Skip if no grads this step

        flat_grad = torch.cat(grads)
        grad_norm = torch.norm(flat_grad, p=2).item()
        grad_var = torch.var(flat_grad).item()

        # Initialize or update running stats
        if self.grad_running_mean is None:
            self.grad_running_mean = flat_grad.clone()
            self.grad_running_mean_squared = flat_grad.clone() ** 2
        else:
            self.grad_running_mean = (self.grad_running_mean * step + flat_grad) / (step + 1)
            self.grad_running_mean_squared = (self.grad_running_mean_squared * step + flat_grad ** 2) / (step + 1)

        # Reduce stats across processes (mean reduction)
        grad_norm_tensor = torch.tensor(grad_norm, device=accelerator.device)
        grad_var_tensor = torch.tensor(grad_var, device=accelerator.device)

        grad_norm_tensor = accelerator.reduce(grad_norm_tensor, reduction="mean")
        grad_var_tensor = accelerator.reduce(grad_var_tensor, reduction="mean")
        flat_grad = accelerator.reduce(flat_grad, reduction="mean")
        self.grad_running_mean = accelerator.reduce(self.grad_running_mean, reduction="mean")
        self.grad_running_mean_squared = accelerator.reduce(self.grad_running_mean_squared, reduction="mean")
        if step + 1 >= 10:
            grad_std = (self.grad_running_mean_squared - self.grad_running_mean ** 2).sqrt()
            lambda_sigma = 3.0
            deviation = (flat_grad - self.grad_running_mean).abs()
            outliers = (deviation > lambda_sigma * grad_std)
            proportion_outliers = outliers.float().mean().item()


        if accelerator.is_main_process:
            info = {
                "grad/post_clip_norm": grad_norm_tensor.item(),
                "grad/variance": grad_var_tensor.item(),
            }
            if step + 1 >= 10:
                info["grad/proportion_spike"] = proportion_outliers
            #print(f"[Step Debug] HF global_step: {state.global_step}, wandb.run.step: {wandb.run.step}")
            wandb.log(info, step=wandb.run.step + 1)
            #print(f"[Step {step + 1}] Pre-clip grad norm: {grad_norm_tensor.item():.4f} | Var: {grad_var_tensor.item():.4f}")
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

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
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

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, cache_dir="/home/wenquan-lu/hf_dataset_cache")

    def format_chat_gsm8k(example):
        return {
            "messages": [
                {"role": "user", "content": example["question"].strip()},
                {"role": "assistant", "content": example["answer"].strip()},
            ]
        }

    # def make_conversation(example, prompt_column="question"):
    #     prompt = []

    #     if training_args.system_prompt is not None:
    #         prompt.append({"role": "system", "content": 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'})

    #     if prompt_column not in example:
    #         raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
    #     solution = extract_hash_answer(example["answer"])
    #     prompt.append({"role": "user", "content": example[prompt_column]})
    #     return {"prompt": prompt, "solution": solution}

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        solution = extract_hash_answer(example["answer"])
        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt, "solution": solution}

    #dataset = dataset.map(make_conversation)

    train_dataset = dataset[script_args.dataset_train_split].map(make_conversation,
                remove_columns=["question", "answer"])
    eval_dataset = dataset[script_args.dataset_test_split].map(make_conversation, 
                remove_columns=["question", "answer"])
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

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # def extract_hash_answer(text: str) -> str | None:
    #     if "####" not in text:
    #         return None
    #     return text.split("####")[1].strip()

    # for split in dataset:
    #     if "messages" in dataset[split].column_names:
    #         dataset[split] = dataset[split].remove_columns("messages")

    #############################
    # Initialize the GRPO trainer
    #############################
    call_backs = get_callbacks(training_args, model_args)
    call_backs.append(GradientMonitorCallback)
    trainer = GRPOEvalTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=(eval_dataset if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=call_backs,
        processing_class=tokenizer,
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
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
