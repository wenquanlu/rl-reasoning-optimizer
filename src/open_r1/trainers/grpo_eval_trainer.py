import logging
import datasets
from .grpo_trainer import GRPOTrainer
import torch

logger = logging.getLogger(__name__)

from transformers.data.data_collator import DataCollatorMixin
from trl.models import unwrap_model_for_generation
from transformers.trainer import EvalLoopOutput
from tqdm import tqdm
from trl.trainer.utils import pad
from trl.extras.profiling import profiling_context
from typing import Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers.utils import is_datasets_available
from dataclasses import dataclass
from transformers import GenerationConfig
#from open_r1.utils.math_eval import remove_boxed, last_boxed_only_string
from vllm import SamplingParams
from open_r1.utils.math_grader import boxed_reward_fn
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
# def extract_boxed_answer(text):
#     boxed = last_boxed_only_string(text)
#     if boxed is None:
#         return None
#     answer = remove_boxed(boxed)
#     answer = answer.replace(",", "").strip()
#     return answer

# more lenient evaluation accept all anchors, e.g., \\boxed, $$$$, last
def reward_style_accuracy(response, gt):
    gold_parsed = parse(
        "\\boxed{" + gt + "}",
        extraction_mode="first_match",
    )
    reward = 0
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            response,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Compute binary rewards if verifiable, `None` otherwise to skip this example
        try:
            reward = int(verify(gold_parsed, answer_parsed))
        except Exception as e:
            print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
    return reward
    

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


@dataclass
class DataCollatorForInferenceVLLM(DataCollatorMixin):
    pad_token_id: int
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples):

        # Pad
        output = {}
        output["prompt"] = [example["prompt"] for example in examples]
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
        if self.use_vllm:
            data_collator = DataCollatorForInferenceVLLM(pad_token_id, None)
        else:
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
        device = self.accelerator.device
        total = 0
        correct = 0
        preds = []
        labels = []
        with torch.no_grad():
            if self.use_vllm:
                if self.vllm_mode == "colocate":
                    for batch in tqdm(dataloader, desc=description):
                        prompts_text = batch['prompt']
                        guided_decoding = None
                        sampling_params = SamplingParams(
                            n=1,  # vLLM on each GPU generates only 1 in colocate mode
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding=guided_decoding,
                        )

                        if self.vllm_tensor_parallel_size > 1:
                            # Gather prompts from all ranks in the TP group and flatten.
                            # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                            orig_size = len(prompts_text)
                            gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                            torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                            all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                            raise Exception
                        else:
                            all_prompts_text = prompts_text

                        with profiling_context(self, "vLLM.generate"):
                            all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                        completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                        if self.vllm_tensor_parallel_size > 1:
                            # Slice completions for this rank within its TP group.
                            # Each rank generates all outputs — we keep only our share.
                            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                            tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                            completion_ids = completion_ids[tp_slice]
                            raise Exception


                        # Pad the completions, and concatenate them with the prompts
                        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                        #completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
                        #prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

                        targets = batch.get("solution", None)

                        for pred, target in zip(completions, targets):
                            # extracted_pred = extract_boxed_answer(pred)
                            # if extracted_pred is not None:
                            #     preds.append(extracted_pred.strip())
                            #     labels.append(target.strip())
                            #     if extracted_pred.strip() == target.strip():
                            #         correct += 1
                            #info, r = boxed_reward_fn(pred, target, fast=False) # boxed_reward has to be from boxed, reward_style would be more lenient
                            r = reward_style_accuracy(pred, target)
                            correct += r
                            total += 1
                else:
                    raise Exception("non-coloate mode not implemented")
            else:
                eval_generation_config = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    do_sample=False,
                    pad_token_id=self.processing_class.pad_token_id,
                    bos_token_id=self.processing_class.bos_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    #cache_implementation=args.cache_implementation,
                )
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
                            # extracted_pred = extract_boxed_answer(pred)
                            # if extracted_pred is not None:
                            #     preds.append(extracted_pred.strip())
                            #     labels.append(target.strip())
                            #     if extracted_pred.strip() == target.strip():
                            #         correct += 1
                            #info, r = boxed_reward_fn(pred, target, fast=False) # boxed_reward has to be from boxed, reward_style would be more lenient
                            r = reward_style_accuracy(pred, target)
                            correct += r
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
            