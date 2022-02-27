from torch.utils.data import Dataset, DataLoader
from utils.dataset_fun import get_dir_pathes, get_file_pathes
import cv2
import os
import numpy as np
import torch

class CellDataset(Dataset):
    def __init__(self, is_traing):
        self.root_dir = 'H:/stem cell/MLP-Mixer/dataset'
        self.is_traning = is_traing
        # 获取全部训练和测试数据的路径
        train_dir_list, test_dir_list = get_dir_pathes()
        # 获取全部图像文件的路径
        self.train_files, self.test_files = get_file_pathes(train_dir_list, test_dir_list)
        self.label_dict = {'a': 0, 'o': 1, 'n': 2}
        self.val_files = self.test_files['a'] + self.test_files['o'] + self.test_files['n']

    def __getitem__(self, item):
        if self.is_traning:
            img_file = self.train_files[item]
        else:
            img_file = self.val_files[item]
        img_arr = cv2.imread(os.path.join(self.root_dir, img_file))
        img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_resize = cv2.resize(img_gray, dsize=(45, 45))
        img_norm = img_resize / 255.0
        img = np.expand_dims(img_norm, 0)
        dir_name = os.path.split(img_file)[0]
        label = self.label_dict[dir_name[0]]
        img_tensor = torch.tensor(img, dtype=torch.float32)
        return img_tensor, label

    def __len__(self):
        length = len(self.train_files) if self.is_traning else len(self.val_files)
        return length

def get_train_loader(batch_size=512):
    train_data = CellDataset(is_traing=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    return train_loader

def get_val_loader(batch_size=512):
    val_data = CellDataset(is_traing=False)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    return val_loader

if __name__ == '__main__':
    cell_data = CellDataset(True)
    train_loader = get_train_loader(batch_size=512)
    print(len(cell_data))

    cell_val_data = CellDataset(False)
    val_loader = get_val_loader(batch_size=512)
    print(len(cell_val_data))