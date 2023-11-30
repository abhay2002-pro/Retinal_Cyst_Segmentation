import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class DatasetClass(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)

    def __len__(self):
        return min(len(os.listdir(os.path.join(self.root, cls))) for cls in self.classes)

    def __getitem__(self, idx):
        class_a, class_b = self.classes
        label = torch.randint(0, 2, (1,)).item()

        folder_a = os.path.join(self.root, class_a)
        folder_b = os.path.join(self.root, class_b)
        images_a = os.listdir(folder_a)
        images_b = os.listdir(folder_b)

        img_a_path = os.path.join(folder_a, images_a[idx])
        img_b_path = os.path.join(folder_b, images_b[idx])

        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b, label
