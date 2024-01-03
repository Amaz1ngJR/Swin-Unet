import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision.transforms import Resize

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def random_crop_images(image1, image2, crop_size=(256, 256), seed=None):
    # 获取图像的大小
    width = 512 
    height = 512

    # 设置种子
    if seed is not None:
        random.seed(seed)

    # 随机生成裁剪框的左上角坐标
    left = random.randint(0, width - crop_size[0])
    top = random.randint(0, height - crop_size[1])

    # 裁剪张量
    cropped_tensor1 = image1[top:top + crop_size[1], left:left + crop_size[0], :]
    cropped_tensor2 = image2[top:top + crop_size[1], left:left + crop_size[0]]

    return cropped_tensor1, cropped_tensor2

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label'] #(512,512,3)

        # 使用相同的种子确保图像和标签在相同的位置被裁剪
        #seed_ = torch.randint(0, 2**32 - 1, (1,))

        #resize_transform = Resize(512) # 设置resize大小

        # 使用 crop 进行裁剪
        # image, label = random_crop_images(image,
        #         label, seed = seed_)
        
        # image = resize_transform(image)
        # label = resize_transform(label)
        #根据一定的概率随机应用旋转和翻转，或者随机旋转
        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        #x, y = image.shape
        x, y, _ = image.shape
        #如果图像的尺寸与期望的输出尺寸不一致，使用 SciPy 中的 zoom 函数进行缩放
        if x != self.output_size[0] or y != self.output_size[1]:
            #image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        #image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        #将 NumPy 数组转换为 PyTorch 张量，并按指定的顺序调整通道维度
        image = torch.from_numpy(image.astype(np.float32)) #(512,512,3)
        image = image.permute(2,0,1) #(3,512,512)
        
        label = torch.from_numpy(label.astype(np.float32))
        
        #sample = {'image': image, 'label': label.long()}#标签被转换为长整型
        sample = {'image': image, 'label': label}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            # vol_name = self.sample_list[idx].strip('\n')
            # filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            # data = h5py.File(filepath)
            # image, label = data['image'][:], data['label'][:]
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            #改，numpy转tensor
            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2,0,1)
            label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
