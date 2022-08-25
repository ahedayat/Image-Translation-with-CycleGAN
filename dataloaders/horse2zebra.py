import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageOps


class Horse2Zebra(Dataset):
    """
        A dataloader for Horse2Zebra Dataset
    """

    def __init__(self, df_path, base_dir, transform=None, mode="test") -> None:
        """
            Parameters:
                - df_path: path to `.csv` file
                - base_dir: base direcory of data
                - transformation: torchvision.transforms that is applied to the input images
        """
        super().__init__()

        assert mode in [
            "train", "val", "test"], "Dataloader mode must be one of this items: ['train', 'val', 'test']"

        self.mode = mode
        self.transform = transform

        self.df = pd.read_csv(df_path)
        self.df.drop(columns=["Unnamed: 0"], inplace=True)

        self.df = self.df[self.df["split"] == self.mode]

        self.df["image_path"] = self.df["image_path"].apply(
            lambda x: os.path.join(base_dir, x))

        df_A = self.df[self.df["domain"] == "A"]
        df_B = self.df[self.df["domain"] == "B"]

        self.files_A = df_A["image_path"].tolist()
        self.files_B = df_B["image_path"].tolist()

    def __getitem__(self, index):
        image_path_A = self.files_A[index % len(self.files_A)]
        image_path_B = self.files_B[index % len(self.files_B)]

        image_B = Image.open(image_path_B).convert("RGB")
        image_A = Image.open(image_path_A).convert("RGB")

        if self.transform is not None:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B

    def __len__(self):
        num_A = len(self.files_A)
        num_B = len(self.files_B)

        return max(num_A, num_B)
