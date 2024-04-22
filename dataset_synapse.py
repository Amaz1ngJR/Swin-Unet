import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2

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
    # 检查裁剪框大小是否超过图像大小
    if crop_size[0] > width or crop_size[1] > height:
        raise ValueError("Crop size is larger than image size")
    # 随机生成裁剪框的左上角坐标
    left = random.randint(0, width - crop_size[0])
    top = random.randint(0, height - crop_size[1])

    # 裁剪张量
    cropped_tensor1 = torch.from_numpy(image1[top:top + crop_size[1], left:left + crop_size[0], :])
    cropped_tensor2 = torch.from_numpy(image2[top:top + crop_size[1], left:left + crop_size[0]])

    return cropped_tensor1, cropped_tensor2 #[256,256,3] [256,256]


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        # 使用 crop 进行裁剪  
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        #image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            
        image = torch.from_numpy(image.astype(np.float32)) #(512,512,3)
        image = image.permute(2,0,1) #(3,512,512)
        
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label} #[3,512,512] [512,512]
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
        # i=idx % 4
        # idx = idx // 4 
        if self.split == "train":
            #idx = idx // 36 
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # if(i == 0):
            #     image = image[:256, :256, :]
            #     label = label[:256, :256]
            # if(i == 1):
            #     image = image[:256, 256:, :]
            #     label = label[:256, 256:]
            # if(i == 2):
            #     image = image[256:, :256, :]
            #     label = label[256:, :256]
            # if(i == 3):
            #     image = image[256:, 256:, :]
            #     label = label[256:, 256:]
            seed_ = torch.randint(0, 2**32 - 1, (1,))
            # print(f"Original image shape: {image_.shape}")
            crop_size=(256,256)
            # image, label=random_crop_images(image_, label_,crop_size, seed_)
            # print(f"Cropped image shape: {image.shape}")

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
            # image = torch.from_numpy(image.astype(np.float32)) #[512,512,3]
            # image = image.permute(2,0,1) #[3,512,512]
            # label = torch.from_numpy(label.astype(np.float32)) #[512,512]
            image  = torch.from_numpy(image).permute(2, 0 ,1)
            label = torch.from_numpy(label)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
