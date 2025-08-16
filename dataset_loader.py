from datasets import load_dataset

def load_fsdd_dataset():
    print("Loading dataset...")
    return load_dataset("mteb/free-spoken-digit-dataset", split="train")
