import datasets
import os

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

if __name__ == "__main__":
    goemotion, yelp = download_datasets()
    print(goemotion.keys())
    print(yelp.keys())



