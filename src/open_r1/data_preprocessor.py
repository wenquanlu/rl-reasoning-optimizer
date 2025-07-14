import re
from datasets import load_dataset
from open_r1.utils import get_model, get_tokenizer
from trl.data_utils import apply_chat_template
from open_r1.utils.math_eval import remove_boxed, last_boxed_only_string


def format_and_truncate_dataset(training_args, tokenizer, eval_dataset):
    eval_dataset = eval_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer}
    )
    
    if training_args.use_vllm:
        def truncate_prompt_for_vllm(example, tokenizer, max_prompt_tokens):
            prompt_text = example["prompt"]
            # Convert to token ids (without padding)
            encoded = tokenizer(prompt_text, truncation=True, add_special_tokens=False)["input_ids"][-max_prompt_tokens:]
            # Decode back to string after truncation
            truncated_prompt = tokenizer.decode(encoded, skip_special_tokens=False)
            example["prompt"] = truncated_prompt
            return example
        eval_dataset = eval_dataset.map(
            truncate_prompt_for_vllm,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_prompt_tokens": training_args.max_prompt_length  # or training_args.max_prompt_length if available
            }
        )

    else:
        def tokenize(example, processing_class):
            prompts_text = example["prompt"]
            prompt_inputs = processing_class(
                text=prompts_text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False
            )
            #prompt_inputs = super()._prepare_inputs(prompt_inputs)
            #prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'][:, -training_args.max_prompt_length :].squeeze(0)
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'][:, -training_args.max_prompt_length :].squeeze(0)
            return prompt_inputs

        eval_dataset = eval_dataset.map(
            tokenize,
            fn_kwargs={
                "processing_class": tokenizer
            },
            remove_columns=["prompt"]
        )
    return eval_dataset

def load_gsm8k_train(script_args, training_args, model_args):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    def extract_hash_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return None
    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

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
    
    return train_dataset

def load_gsm8k_eval(script_args, training_args, model_args, tokenizer):
    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    def extract_hash_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return None
    
    # Format into conversation
    def make_conversation(example, prompt_column="question"):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        solution = extract_hash_answer(example["answer"])
        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt, "solution": solution}

    eval_dataset = load_dataset('openai/gsm8k', name='main')['test'].map(make_conversation, 
                remove_columns=["question", "answer"])
    
    if training_args.eval_strategy != "no":
        eval_dataset = format_and_truncate_dataset(training_args, tokenizer, eval_dataset)
    return eval_dataset
    

def load_math_train(script_args, training_args, model_args):
    # def extract_boxed_answer(text):
    #     boxed = last_boxed_only_string(text)
    #     if boxed is None:
    #         return None
    #     answer = remove_boxed(boxed)
    #     answer = answer.strip()
    #     print(answer)
    #     return answer

    # Load the dataset
    train_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)[script_args.dataset_train_split]

        # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}


    train_dataset = train_dataset.map(make_conversation,
                remove_columns=["problem", "level", "type"])
    
    return train_dataset


def load_math500_eval(script_args, training_args, model_args, tokenizer):

    def make_eval_conversation(example, prompt_column="problem"):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        prompt.append({"role": "user", "content": example[prompt_column]})
        solution = example["answer"]
        return {"prompt": prompt, "solution": solution}

    eval_dataset = load_dataset("HuggingFaceH4/MATH-500", name="default")["test"]
    eval_dataset = eval_dataset.map(make_eval_conversation, 
                remove_columns=["problem", "answer", "subject", "level", "unique_id"])

    if training_args.eval_strategy != "no":
        eval_dataset = format_and_truncate_dataset(training_args, tokenizer, eval_dataset)
    return eval_dataset


def load_aime24_eval(script_args, training_args, model_args, tokenizer):
    def make_eval_conversation(example, prompt_column="Problem"):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        prompt.append({"role": "user", "content": example[prompt_column]})
        solution = str(example["Answer"])
        return {"prompt": prompt, "solution": solution}

    eval_dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    eval_dataset = eval_dataset.map(make_eval_conversation, 
                remove_columns=["ID", "Problem", "Solution", "Answer"])

    if training_args.eval_strategy != "no":
        eval_dataset = format_and_truncate_dataset(training_args, tokenizer, eval_dataset)
    return eval_dataset


def load_amc_eval(script_args, training_args, model_args, tokenizer):
    def make_eval_conversation(example, prompt_column="problem"):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        prompt.append({"role": "user", "content": example[prompt_column]})
        solution = example["answer"]
        solution = str(int(solution)) if solution == int(solution) else str(solution)
        return {"prompt": prompt, "solution": solution}

    eval_dataset = load_dataset("AI-MO/aimo-validation-amc")["train"]
    eval_dataset = eval_dataset.map(make_eval_conversation, 
                remove_columns=["id", "problem", "answer", "url"])

    if training_args.eval_strategy != "no":
        eval_dataset = format_and_truncate_dataset(training_args, tokenizer, eval_dataset)
    return eval_dataset



def load_aime25_eval(script_args, training_args, model_args, tokenizer):
    def make_eval_conversation(example, prompt_column="problem"):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        prompt.append({"role": "user", "content": example[prompt_column]})
        solution = example["answer"]
        return {"prompt": prompt, "solution": solution}

    eval_dataset = load_dataset("yentinglin/aime_2025", name="default")["train"]
    eval_dataset = eval_dataset.map(make_eval_conversation, 
                remove_columns=["id", "problem", "answer", "url", "year", "__index_level_0__"])

    if training_args.eval_strategy != "no":
        eval_dataset = format_and_truncate_dataset(training_args, tokenizer, eval_dataset)
    return eval_dataset


