import datasets
import os
from transformers import AutoTokenizer

def download_datasets():
    goemotion = datasets.load_dataset("go_emotions", "simplified")
    yelp = datasets.load_dataset("Yelp/yelp_review_full")
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/goemotion.hf"):
        print("Saving goemotion to disk")
        goemotion.save_to_disk("data/goemotion.hf")
    else:
        print("Goemotion already exists on disk")
    if not os.path.exists("data/yelp.hf"):
        print("Saving yelp to disk")
        yelp.save_to_disk("data/yelp.hf")
    else:
        print("Yelp already exists on disk")
    
    return goemotion, yelp

def save_tokenized_datasets(dataset, tokenizer, path):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.save_to_disk(path)
    return tokenized_dataset

if __name__ == "__main__":
    goemotion, yelp = download_datasets()
    print(goemotion.keys())
    print(yelp.keys())
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(tokenizer.vocab_size)
    if not os.path.exists("data/goemotion_tokenized.hf"):
        goemotion = save_tokenized_datasets(goemotion, tokenizer, "data/goemotion_tokenized.hf")
    goemotion = save_tokenized_datasets(goemotion, tokenizer, "data/goemotion_tokenized.hf")
    if not os.path.exists("data/yelp_tokenized.hf"):
        yelp = save_tokenized_datasets(yelp, tokenizer, "data/yelp_tokenized.hf")
    print(goemotion.keys()) 
    print(yelp.keys())



