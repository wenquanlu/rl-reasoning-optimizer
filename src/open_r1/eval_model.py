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
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser, get_peft_config
import torch
from open_r1.data_preprocessor import load_math_train, load_gsm8k_train, load_gsm8k_eval, load_math500_eval, load_aime24_eval, load_aime25_eval, load_amc_eval
logger = logging.getLogger(__name__)

from transformers import TrainerCallback
import wandb
import re
from open_r1.trainers.grpo_eval_trainer import GRPOEvalTrainer

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_hash_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return None


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
    
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.padding_side = 'left'
    print("tokenizer padding side:", tokenizer.padding_side )
    print("eval_strategy", training_args.eval_strategy)

    if script_args.dataset_name == "openai/gsm8k":
        train_dataset = load_gsm8k_train(script_args, training_args, model_args)
    elif script_args.dataset_name == "DigitalLearningGmbH/MATH-lighteval":
        train_dataset = load_math_train(script_args, training_args, model_args)
    
    eval_loaders = {
        "gsm8k": load_gsm8k_eval,
        "math500": load_math500_eval,
        "aime24": load_aime24_eval,
        "amc": load_amc_eval,
        "aime25": load_aime25_eval
    }
    eval_dataset = {}
    if training_args.eval_dataset_names:
        for eval_dataset_name in training_args.eval_dataset_names:
            if eval_dataset_name in eval_loaders:
                loader = eval_loaders[eval_dataset_name]
                eval_dataset[eval_dataset_name] = loader(script_args, training_args, script_args, tokenizer)

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

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        #metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
