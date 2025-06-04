from datasets import load_dataset
ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")
print(len(ds["train"]))