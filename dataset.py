import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class PolypDataset(Dataset):
    def __init__(self, path, image_size, val_dataset='CVC-ClinicDB', train=True):
        self.image_size = image_size
        self.train = train
        if train:
            path = os.path.join(path, 'TrainDataset')
        else:
            path = os.path.join(path, 'TestDataset', val_dataset)
        self.image_paths = []
        self.mask_paths = []
        image_folder = os.path.join(path, 'images')
        mask_folder = os.path.join(path, 'masks')
        for sample_name in os.listdir(image_folder):            
            self.image_paths.append(os.path.join(image_folder, sample_name))
            self.mask_paths.append(os.path.join(mask_folder, sample_name))

    
    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index])
        image = image.convert('RGB')
        numpy_image = np.array(image)
        mask = Image.open(self.mask_paths[index])
        mask = mask.convert('L')
        numpy_mask = np.array(mask)
        numpy_mask = np.where(numpy_mask > 0, 1, 0).astype(np.float32)

        if self.train:
            transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=90, p=0.5),

                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),

                A.RandomResizedCrop(height=self.image_size[0], width=self.image_size[1], scale=(0.8, 1.0), p=0.5),

                A.GaussianBlur(p=0.3),
                A.GaussNoise(p=0.3),

                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
        transformed = transform(image=numpy_image, mask=numpy_mask)
        image = transformed['image']
        mask = transformed['mask']

        return (image, mask)

    def __len__(self):
        return len(self.image_paths)


# class ISICDataset(Dataset):
#     def __init__(self, path, image_size, train=True, with_name=False):
#         self.image_size = image_size
#         self.train = train
#         self.with_name = with_name
#         if train:
#             path = os.path.join(path, 'train')
#         else:
#             path = os.path.join(path, 'val')
#         self.sample_names = []
#         self.image_paths = []
#         self.mask_paths = []
#         image_folder = os.path.join(path, 'images')
#         mask_folder = os.path.join(path, 'masks')
#         for sample_name in os.listdir(image_folder):            
#             self.sample_names.append(sample_name)
#             self.image_paths.append(os.path.join(image_folder, sample_name))
#             self.mask_paths.append(os.path.join(mask_folder, sample_name.replace('.jpg', '_segmentation.png')))

    
#     def __getitem__(self, index: int):
#         image = Image.open(self.image_paths[index])
#         image = image.convert('RGB')
#         numpy_image = np.array(image)
#         mask = Image.open(self.mask_paths[index])
#         mask = mask.convert('L')
#         numpy_mask = np.array(mask)
#         numpy_mask = np.where(numpy_mask > 0, 1, 0).astype(np.float32)

#         if self.train:
#             transform = A.Compose([
#                 A.Resize(height=self.image_size[0], width=self.image_size[1]),
#                 A.HorizontalFlip(),
#                 A.VerticalFlip(),
#                 A.Rotate(limit=90),
#                 A.RandomCrop(height=self.image_size[0], width=self.image_size[0]),
#                 A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#                 ToTensorV2(),
#             ])
#         else:
#             transform = A.Compose([
#                 A.Resize(height=self.image_size[0], width=self.image_size[1]),
#                 A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#                 ToTensorV2(),
#             ])
#         transformed = transform(image=numpy_image, mask=numpy_mask)
#         image = transformed['image']
#         mask = transformed['mask']

#         if self.with_name:
#             name = self.sample_names[index]
#             return (name, image, mask)
#         else:
#             return (image, mask)

#     def __len__(self):
#         return len(self.image_paths)
    

# import h5py

# class Synapse_dataset(Dataset):
#     def __init__(self, path, image_size, train=True):
#         self.sample_list = open(os.path.join(path, 'lists', 'lists_Synapse', f"{'train' if train else 'test_vol'}.txt")).readlines()
#         self.data_dir = os.path.join(path, 'train_npz' if train else 'test_vol_h5')
#         self.image_size = image_size
#         self.train = train

#     def __getitem__(self, idx):
#         if self.train == True:
#             slice_name = self.sample_list[idx].strip('\n')
#             data_path = os.path.join(self.data_dir, slice_name+'.npz')
#             data = np.load(data_path)
#             numpy_image, numpy_mask = data['image'], data['label']
#             transform = A.Compose([
#                 A.Resize(height=self.image_size[0], width=self.image_size[1]),
#                 A.HorizontalFlip(),
#                 A.VerticalFlip(),
#                 A.Rotate(limit=90),
#                 A.RandomCrop(height=self.image_size[0], width=self.image_size[0]),
#                 ToTensorV2(),
#             ])
#             transformed = transform(image=numpy_image, mask=numpy_mask)
#             image = transformed['image'].float().repeat(3, 1, 1)
#             mask = transformed['mask'].long()
#             return (image, mask)
#         else:
#             vol_name = self.sample_list[idx].strip('\n')
#             filepath = os.path.join(self.data_dir, f"{vol_name}.npy.h5")
#             data = h5py.File(filepath)
#             numpy_volumn, numpy_label = data['image'][:], data['label'][:]
#             transform = A.Compose([
#                 A.Resize(height=self.image_size[0], width=self.image_size[1]),
#                 A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#                 ToTensorV2(),
#             ])
#             num_slices = numpy_volumn.shape[0]
#             volumn = torch.zeros((num_slices, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32)
#             label = torch.zeros((num_slices, self.image_size[0], self.image_size[1]), dtype=torch.long)
#             for i in range(num_slices):
#                 numpy_slice = numpy_volumn[i]
#                 transformed = transform(image=numpy_slice, mask=numpy_label[i])
#                 volumn[i] = transformed['image'].repeat(3, 1, 1)
#                 label[i] = transformed['mask']
#             return volumn, label
        
#     def __len__(self):
#         return len(self.sample_list)