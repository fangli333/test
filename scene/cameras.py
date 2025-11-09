#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, focal2fov, getWorld2View2_GPU, getProjectionMatrix_GPU, quater2rotation
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, PILtoTorch

class Camera(nn.Module):

    share_focal = None
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0, mask = None, depth=None, pt = None, index = None):
        super(Camera, self).__init__()
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.uid = uid
        self.colmap_id = colmap_id
        self.original_image = PILtoTorch(image, (int(image.size[0] / scale), int(image.size[1] / scale))).clamp(0.0, 1.0)[:3,:,:].to(self.data_device)
        self.time = torch.Tensor(np.array([time])).to(self.data_device)
        self.image_width = self.original_image.shape[2]  #480
        self.image_height = self.original_image.shape[1]   #270
        self.image_name = image_name

        self.zfar = torch.tensor(100.0).to(self.data_device)
        self.znear = torch.tensor(0.01).to(self.data_device)
        self.trans = torch.tensor(trans).to(self.data_device)
        self.scale = torch.tensor(scale).to(self.data_device)

        self.pt = torch.Tensor(pt).to(self.data_device) if pt is not None else None
        self.mask = torch.Tensor(gt_alpha_mask).to(self.data_device) if gt_alpha_mask is not None else None
        self.index = torch.tensor(index, dtype=torch.long).to(self.data_device) if index is not None else None

        if self.index != None:
            self.quaternion = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype = torch.float32).to(self.data_device))
            self.T = nn.Parameter(torch.tensor(T, dtype = torch.float32).to(self.data_device))
        else:
            self.quaternion = None
            self.R = torch.tensor(R).to(self.data_device)
            self.T = torch.tensor(T).to(self.data_device)

        if self.index == None:
            focalx = self.image_width / (2 * torch.tan(torch.tensor(FoVx).to(self.data_device) / 2))
            focaly = self.image_height / (2 * torch.tan(torch.tensor(FoVy).to(self.data_device) / 2))
            focal = (focalx + focaly) / 2
            self.focal = focal

        else:
            if Camera.share_focal is None:
                Camera.share_focal = nn.Parameter(torch.tensor((self.image_height + self.image_width) / 2, dtype = torch.float32).cuda())
            self.focal = Camera.share_focal


    def wp_compute(self):
        if self.quaternion != None:
            self.R = quater2rotation(self.quaternion).transpose(0, 1)
        self.FoVx = 2 * torch.atan(self.image_width / (2 * self.focal))
        self.FoVy = 2 * torch.atan(self.image_height / (2 * self.focal))
        self.world_view_transform = getWorld2View2_GPU(R = self.R, t = self.T, translate = self.trans, scale = self.scale, device = self.data_device).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix_GPU(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, device = self.data_device).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


    def training_setup(self, training_args):
        self.spatial_lr_scale = 5

        l = [
            {'params': [self.quaternion], 'lr': training_args.quaternion_lr_init * self.spatial_lr_scale, "name": "quaternion"},
            {'params': [self.T], 'lr': training_args.t_lr_init * self.spatial_lr_scale, "name": "T"},
            {'params': [self.focal], 'lr': training_args.focal_lr_init * self.spatial_lr_scale, "name": "focal"}
        ]
        #momentum optimzation method in Adam, also changes in gaussian and deformation
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=(0.9, 0.999))
        self.quaternion_scheduler_args = get_expon_lr_func(lr_init=training_args.quaternion_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.quaternion_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.quaternion_lr_delay_mult,
                                                    max_steps=training_args.quaternion_lr_max_steps)
        
        self.t_scheduler_args = get_expon_lr_func(lr_init=training_args.t_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.t_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.t_lr_delay_mult,
                                                    max_steps=training_args.t_lr_max_steps)
        
        self.focal_scheduler_args = get_expon_lr_func(lr_init=training_args.focal_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.focal_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.focal_lr_delay_mult,
                                                    max_steps=training_args.focal_lr_max_steps)
    
    
    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == 'quaternion':
                lr = self.quaternion_scheduler_args(iteration)
                param_group['lr'] = lr
            
            if param_group["name"] == 'T':
                lr = self.t_scheduler_args(iteration)
                param_group['lr'] = lr

            if param_group["name"] == 'focal':
                lr = self.focal_scheduler_args(iteration)
                param_group['lr'] = lr






class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

