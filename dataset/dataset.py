import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
class NERDataset(Dataset):
    def __init__(self, data_path, tag_path, char2id, tag2id):
        self.samples = []
        self.char2id = char2id
        self.tag2id = tag2id

        with open(data_path, "r", encoding="utf-8") as f_data, \
             open(tag_path, "r", encoding="utf-8") as f_tag:
            
            for line_data, line_tag in zip(f_data, f_tag):
                chars = line_data.strip().split()
                tags = line_tag.strip().split()

                if len(chars) != len(tags):
                    raise ValueError(f"字符数和标签数不一致: {chars} || {tags}")

                input_ids = [char2id.get(ch, char2id["<UNK>"]) for ch in chars]
                label_ids = [tag2id[tag] for tag in tags]

                self.samples.append({
                    "input_ids": input_ids,
                    "label_ids": label_ids,
                    "length": len(input_ids)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "label_ids": torch.tensor(sample["label_ids"], dtype=torch.long),
            "length": sample["length"]
        }
        
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    label_ids = [item["label_ids"] for item in batch]
    lengths = [item["length"] for item in batch]

    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)

    attention_mask = (input_ids != 0).long()

    return {
        "input_ids": input_ids,
        "label_ids": label_ids,
        "attention_mask": attention_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long)
    }