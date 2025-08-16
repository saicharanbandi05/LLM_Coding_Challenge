from datasets import load_dataset

def load_fsdd_dataset():
    """
    Loads the Free Spoken Digit Dataset (FSDD) from Hugging Face.
    """
    print("Loading dataset...")
    return load_dataset("mteb/free-spoken-digit-dataset", split="train")
