""" train and test dataset

author jundewu
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from func_2d.utils import random_click


class REFUGE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):
        self.data_path = data_path
        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode + '-400')) if f.is_dir()]
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):

        """Get the images"""
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        # raw image and raters path
        img_path = os.path.join(subfolder, name + '_cropped.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '_cropped.jpg') for i in range(1, 8)]

        # img_path = os.path.join(subfolder, name + '.jpg')
        # multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]

        # raw image and rater images
        img = Image.open(img_path).convert('RGB')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]

        # apply transform
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >=0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            torch.set_rng_state(state)

        # find init click and apply majority vote
        if self.prompt == 'click':

            point_label_cup, pt_cup = random_click(np.array((multi_rater_cup.mean(axis=0)).squeeze(0)), point_label = 1)
            
            selected_rater_mask_cup_ori = multi_rater_cup.mean(axis=0)
            selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 


            selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
            selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


            # # Or use any specific rater as GT
            # point_label_cup, pt_cup = random_click(np.array(multi_rater_cup[0, :, :, :].squeeze(0)), point_label = 1)
            # selected_rater_mask_cup_ori = multi_rater_cup[0,:,:,:]
            # selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 

            # selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
            # selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'multi_rater': multi_rater_cup, 
            'p_label': point_label_cup,
            'pt':pt_cup, 
            'mask': selected_rater_mask_cup, 
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict':image_meta_dict,
        }


