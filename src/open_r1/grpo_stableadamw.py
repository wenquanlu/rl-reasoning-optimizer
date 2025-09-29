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
from trl import ModelConfig, TrlParser, get_peft_config
from open_r1.data_preprocessor import load_math_train, load_gsm8k_train, load_gsm8k_eval, load_math500_eval, load_aime24_eval, load_amc_eval, load_aime25_eval, load_minerva_eval, load_olympiad_eval
logger = logging.getLogger(__name__)


from open_r1.trainers.grpo_eval_trainer import GRPOEvalTrainer
from open_r1.custom_callbacks import OptimStateCleanupCallback, ForceEvalCallback


import sys
import os
from optimi import StableAdamW
from open_r1.utils.train_utils import get_decay_parameter_names


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
        "aime25": load_aime25_eval,
        "aime24_avg8": load_aime24_eval,
        "aime25_avg8": load_aime25_eval,
        "amc_avg8": load_amc_eval,
        "minerva": load_minerva_eval,
        "olympiad": load_olympiad_eval
    }
    eval_dataset = {}
    if training_args.eval_dataset_names:
        for eval_dataset_name in training_args.eval_dataset_names:
            if eval_dataset_name in eval_loaders:
                loader = eval_loaders[eval_dataset_name]
                eval_dataset[eval_dataset_name] = loader(script_args, training_args, model_args, tokenizer)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)
    decay_parameters = get_decay_parameter_names(model)
    decay_value = getattr(training_args, "weight_decay", 0.0)
    learning_rate_value = getattr(training_args, "learning_rate", 2.0e-05)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": decay_value,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = StableAdamW(optimizer_grouped_parameters, lr=learning_rate_value, weight_decay=0)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)


    #############################
    # Initialize the GRPO trainer
    #############################
    call_backs = get_callbacks(training_args, model_args)
    call_backs.append(OptimStateCleanupCallback)
    call_backs.append(ForceEvalCallback)
    trainer = GRPOEvalTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=(eval_dataset if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=call_backs,
        processing_class=tokenizer,
        optimizers=(optimizer, None)
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
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
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
        #metrics["eval_samples"] = len(eval_dataset)
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
