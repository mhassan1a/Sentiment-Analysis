from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_from_disk, dataset_dict

class GoEmotionsDataset(Dataset):
    def __init__(self, split):
        datasets = load_from_disk("data/goemotion_tokenized.hf")
        
        self.dataset = datasets[split]
        self.encoding = self.dataset['input_ids']
        self.labels = self.dataset['labels']
        self.mask = self.dataset['attention_mask']

        self.flat_encoding = []
        self.flat_labels = []
        self.flat_mask = []

        for i in range(len(self.encoding)):
            for label in self.labels[i]:
                self.flat_encoding.append(self.encoding[i])
                self.flat_labels.append(label)
                self.flat_mask.append(self.mask[i])

    def __getitem__(self, idx):
        token = torch.tensor(self.flat_encoding[idx], dtype=torch.long)
        label = torch.tensor(self.flat_labels[idx], dtype=torch.long)
        mask = torch.tensor(self.flat_mask[idx], dtype=torch.bool)
        return (token, label, mask)

    def __len__(self):
        return len(self.encoding)

   
        
if __name__ == "__main__":
    goemotion_train = GoEmotionsDataset("train")
    goemotion_train_loader = DataLoader(goemotion_train, batch_size=32, shuffle=True)
    print(f"Length of train dataset: {len(goemotion_train)}")
    for input, label, mask in goemotion_train_loader:
        print(input.size(), label.size(), mask.size())
        break