import os
from torch.utils.data import Dataset

import utils

class COCODataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.len = self.get_len()

    def get_len(self):
        len = 0
        for _ in os.scandir(self.img_dir):
            len += 1
        return len

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # Note that the file name in COCO dataset is renamed to 1.jpg, 2.jpg, etc 
        file_path = os.path.join(self.img_dir, f'{index}.jpg')

        return utils.read_img(file_path)