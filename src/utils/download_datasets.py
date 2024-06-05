import datasets
import os

def download_datasets():
    goemotion = datasets.load_dataset("go_emotions", "simplified")
    yelp = datasets.load_dataset("Yelp/yelp_review_full")
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/goemotion.hf"):
        goemotion.save_to_disk("data/goemotion.hf")
    if not os.path.exists("data/yelp.hf"):
        yelp.save_to_disk("data/yelp.hf")
    
    return goemotion, yelp

if __name__ == "__main__":
    goemotion, yelp = download_datasets()
    print(goemotion.keys())
    print(yelp.keys())



