from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_from_disk

class YelpDataset(Dataset):
    def __init__(self,  split):
        datasets = load_from_disk("data/yelp_tokenized.hf")
        self.dataset = datasets[split]
        print(self.dataset.column_names)
        self.encoding = self.dataset['input_ids']
        print(self.encoding[0])
        self.labels = self.dataset['label']
        print(self.labels[0])

    def __getitem__(self, idx):
        token = torch.tensor(self.encoding[idx])
        label = torch.tensor(self.labels[idx])
        return token, label

    def __len__(self):
        return len(self.labels)
    
    
if __name__ == "__main__":
    yelp_train = YelpDataset("train")
    yelp_train_loader = DataLoader(yelp_train, batch_size=2, shuffle=False)
    for batch in yelp_train_loader:
        print(batch)
        break
    