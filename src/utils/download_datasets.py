import datasets
import os
from transformers import AutoTokenizer

def download_datasets():
    emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 
        'remorse', 'sadness', 'surprise', 'neutral'
    ]

    goemotion = datasets.load_dataset("go_emotions", "raw")

    # Ensure 'goemotion' dataset has columns corresponding to 'emotions'
    def get_labels(example):
        return {"labels": [example[emotion] for emotion in emotions]}

    goemotion = goemotion.map(get_labels)
    
    # Filter out rows where no label is 1
    goemotion = goemotion.filter(lambda x: 1 in x["labels"])
    
    # Map to get the index of the label with value 1
    goemotion = goemotion.map(lambda x: {"labels": x["labels"].index(1)})
    print(goemotion['train'][0:5])
    yelp = datasets.load_dataset("yelp_review_full")

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
    else:
        goemotion = datasets.load_from_disk("data/goemotion_tokenized.hf")
    
    if not os.path.exists("data/yelp_tokenized.hf"):
        yelp = save_tokenized_datasets(yelp, tokenizer, "data/yelp_tokenized.hf")
    else:
        yelp = datasets.load_from_disk("data/yelp_tokenized.hf")
    
    print(goemotion.keys()) 
    print(yelp.keys())
