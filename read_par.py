from datasets import load_dataset

# Load from local Parquet file
dataset = load_dataset("parquet", data_files="train.parquet")
print(next(iter(dataset['train'])))
print(next(iter(dataset['train'])))
print(next(iter(dataset['train'])))