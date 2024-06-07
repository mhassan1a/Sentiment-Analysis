from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_from_disk, dataset_dict

class GoEmotionsDataset(Dataset):
    def __init__(self, split):
        datasets = load_from_disk("data/goemotion_tokenized.hf")
        self.dataset = datasets[split]
        self.encoding = self.dataset['input_ids']
        self.labels = self.dataset['labels']

    def __getitem__(self, idx):
        token = torch.tensor(self.encoding[idx])
        label = torch.tensor(self.labels[idx][0])
        return (token, label)

    def __len__(self):
        return len(self.encoding)

   
        
if __name__ == "__main__":
    goemotion_train = GoEmotionsDataset("train")
    goemotion_train_loader = DataLoader(goemotion_train, batch_size=32, shuffle=True)
    print(f"Length of train dataset: {len(goemotion_train[:100])}")
    for input, label in goemotion_train_loader:
        print(input.size(), label.size())
        break