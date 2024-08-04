import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
import nibabel as nib
from torch.utils.data import Dataset

from utils import random_click


class AMOS(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', val_seed=None):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))
        
        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        # num_frames = []
        # for name in self.name_list:
        #     num_frames.append(len(os.listdir(os.path.join(data_path, mode, 'image', name))))
        # max_frame = max(num_frames)
        # if mode == "Test":
        #     if val_seed is None:
        #         self.val_seed = None
        #     else:
        #         self.val_seed = (val_seed * np.arange(max_frame * len(self.name_list)).reshape(len(self.name_list), max_frame)).tolist()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        # if self.mode == 'Training':
        #     val_seed = None
        # else:
        #     val_seed = self.val_seed[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        img_3d = nib.load(img_path)
        data_img_3d = img_3d.get_fdata()
        seg_3d = nib.load(mask_path)
        data_seg_3d = seg_3d.get_fdata()
        num_frame = data_img_3d.shape[-1]
        img_tensor = torch.tensor(num_frame, 1, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}

        for frame_index in range(num_frame):
            img = data_img_3d[..., frame_index]
            mask = data_seg_3d[..., frame_index]
            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}
            diff_obj_pt_dict = {}
            diff_obj_point_label_dict = {}
            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask.resize(newsize)
                if self.transform_msk:
                    obj_mask = self.transform_msk(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask), point_label, seed=None)
            
            if self.transform:
                # state = torch.get_rng_state()
                img = self.transform(img)
                # torch.set_rng_state(state)


                

            img_tensor[frame_index, 0] = img
            mask_dict[frame_index] = diff_obj_mask_dict
            pt_dict[frame_index] = diff_obj_pt_dict
            point_label_dict[frame_index] = diff_obj_point_label_dict


        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img_tensor,
            'label': mask_dict,
            'p_label':point_label_dict,
            'pt':pt_dict,
            'image_meta_dict':image_meta_dict,
        }