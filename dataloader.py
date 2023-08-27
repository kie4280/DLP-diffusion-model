import json
from torch.utils.data import Dataset
import torch
from typing import List, Tuple
import torchvision
import glob
import pathlib


class LabelTransformer:
    def __init__(self, filename: str = "objects.json", label_num: int = 24) -> None:
        with open(filename) as f:
            self.labels = json.load(f)
        self.label_num = label_num

    def transform(self, x: str):
        return self.labels[x]


class Training_dataset(Dataset):
    def __init__(
        self, filename: str = "train.json", dataset_folder: str = "iclevr/"
    ) -> None:
        super().__init__()
        with open(filename) as f:
            self.dataset_dict = json.load(f)
        self.data_folder = dataset_folder
        self.translator = LabelTransformer()

    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, index) -> Tuple[torch.Tensor, List[str]]:
        key = list(self.dataset_dict.keys())[index]
        filepath = f"{self.data_folder}/{key}"
        img = (
            torchvision.io.read_image(filepath, torchvision.io.ImageReadMode.RGB).to(
                dtype=torch.float
            )
            / 255.0
        )
        labels = torch.tensor(
            tuple(self.translator.transform(x) for x in self.dataset_dict[key]),
            dtype=torch.long,
        )
        labels = torch.nn.functional.one_hot(
            labels, num_classes=self.translator.label_num
        )
        labels = torch.sum(labels, dim=0, keepdim=True).to(dtype=torch.float)

        return (img, labels)


class Testing_dataset(Dataset):
    def __init__(self, filename: str) -> None:
        super().__init__()
        with open(filename) as f:
            self.dataset = json.load(f)
        self.translator = LabelTransformer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> List[str]:
        labels = torch.tensor(
            tuple(self.translator.transform(x) for x in self.dataset[index]),
            dtype=torch.long,
        )
        labels = torch.nn.functional.one_hot(
            labels, num_classes=self.translator.label_num
        )
        labels = torch.sum(labels, dim=0, keepdim=True).to(dtype=torch.float)
        return labels
