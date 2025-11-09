import torch
import numpy as np
from PIL import Image, ImageDraw
from skimage import data, color
# from scipy.ndimage import gaussian_gradient_magnitude, maximum_filter, minimum_filter, sobel
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import json
import matplotlib.path as mpl_path
import cv2
from collections import defaultdict
import torch.nn.functional as F 
import time

##########PARAMETERS TO BE DONE##########
    '''
    threshold = 100 --- By Default
    patch_size = 12 or 24 --- By Default
    select_bin, add_bin --- We recommend to try full-sequence length first. Please reduce the frame count up to your GPU memory.

    '''
data = f'DATANAME'
threshold = f'{100}'
patch_size = f'{12}'

device = 'cuda'
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
save_path = f'YOURSAVEFOLDERPATH/{data}'
folder_path = f'YOURDATAFOLDERPATH/{data}'
os.makedirs(save_path, exist_ok=True)


filess = os.listdir(folder_path)
filess.sort()
files = [item for item in filess if item.lower().endswith(('.png', '.jpg', '.jpeg'))]

images = [iio.imread(os.path.join(folder_path, file)) for file in files]
images = np.array(images) 
images = torch.tensor(images, device='cuda')

select_bin = f'{images.shape[0]}'
add_bin = f'{images.shape[0]}'

num_global = 0
point_pos = torch.ones((images.shape[0], threshold, 2)).to(device) * (-1)
point_ind = torch.ones((images.shape[0], threshold)).to(device) * (-1)

def rgb_to_grayscale(frame):
    frame = frame.to(torch.float64) / 255.0
    r = frame[:, :, 0]
    g = frame[:, :, 1]
    b = frame[:, :, 2]
    grayframe = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return grayframe

def high_texture_patch(patch_size, grayframe):
    patches = grayframe.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    n_patches_height = patches.size(0)
    n_patches_width = patches.size(1)
    patches = patches.contiguous().view(n_patches_height, n_patches_width, patch_size, patch_size)
    variances = torch.var(patches.view(n_patches_height * n_patches_width, -1), dim=1)
    variances = variances.view(n_patches_height, n_patches_width)
    variance_threshold = 0.01 * torch.max(variances)
    texture_map = variances > variance_threshold

    return patches, texture_map

def calculate_gradient(frame):
    frame = frame.cpu().numpy()
    grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = torch.tensor(gradient_magnitude, device='cuda')

    return gradient_magnitude

def location_pool(frame, device, patch_size):
    grayframe = rgb_to_grayscale(frame)
    patches, texture_map = high_texture_patch(patch_size, grayframe)
    gradient_map = torch.zeros_like(grayframe, dtype=torch.uint8, device=device)

    # Iterate through each patch
    for i in range(texture_map.shape[0]):
        for j in range(texture_map.shape[1]):
            if texture_map[i, j]:  # Only consider rich-texture patches
                patch = patches[i, j]
                gradient_magnitude = calculate_gradient(patch)
                max_idx = torch.unravel_index(torch.argmax(gradient_magnitude), gradient_magnitude.shape)
                global_max_x = i * patch_size + max_idx[0].item()
                global_max_y = j * patch_size + max_idx[1].item()
                gradient_map[global_max_x, global_max_y] = 1

    loca_pool = torch.nonzero(gradient_map == 1)

    return loca_pool

def assign_index(credit, inter, threshold):
    index_all = torch.ones(credit.shape[0], 1).to(device) * (-1)
    index_init = 0
    for m in range(credit.shape[0]):
        if credit[m] == 1:
            index_all[m] = index_init
            index_init += 1

    num_old = threshold - torch.count_nonzero(credit)
    where_old, _ = torch.where((inter == 1.0) & (credit != 1.0))
    where_old = where_old[torch.randperm(where_old.size(0))[:num_old]]
    for n in range(where_old.shape[0]):
        index_all[where_old[n]] = index_init
        index_init += 1

    assert index_all.max() == threshold - 1
    where_new, _ = torch.where(credit == 1.0)

    return index_all.squeeze(), where_new, where_old


def less_than_one_per_patch(credit, loca_ori, video, patch_size, current_frame_i, points_live):
    loca = loca_ori[torch.nonzero(credit, as_tuple=True)[0]]
    ori_indice0 = torch.nonzero(credit, as_tuple=True)[0]

    assert loca.shape[0] == ori_indice0.shape[0]

    grayframe = rgb_to_grayscale(video[0, current_frame_i, :, :, :].permute(1, 2, 0))
    gradient_magnitude = calculate_gradient(grayframe)

    point_patches_live = [(int(h // patch_size), int(w // patch_size)) for h, w in points_live]
    point_patches0 = [(int(h // patch_size), int(w // patch_size)) for h, w in loca]
    
    assert len(point_patches0) == ori_indice0.shape[0] == loca.shape[0]
    
    h_patch_max = grayframe.shape[0] // patch_size
    w_patch_max = grayframe.shape[1] // patch_size
    point_patches_live = [t for t in point_patches_live if not (t[0] >= h_patch_max or t[1] >= w_patch_max)]
 
    point_patches = []
    ori_indice = []
    for w in range(len(point_patches0)):
        if not (point_patches0[w][0] >= h_patch_max or point_patches0[w][1] >=  w_patch_max):
            point_patches.append(point_patches0[w])
            ori_indice.append(ori_indice0[w])
    ori_indice = torch.tensor(ori_indice, device='cuda')
    
    points_in_patch = defaultdict(list)
    for i, patch in enumerate(point_patches):
        points_in_patch[patch].append(i)
    for patch, indices in points_in_patch.items():

        if patch in point_patches_live:
            credit[ori_indice[indices]] = 0         
        elif len(indices) > 1:
            patch_gradients = torch.tensor([gradient_magnitude[int(loca[i][0]), int(loca[i][1])] for i in indices], device = 'cuda')
            mask = torch.ones(len(indices), dtype=torch.bool, device='cuda')
            mask[torch.argmax(patch_gradients)] = False
            reduce_indices = ori_indice[indices][mask]
            credit[reduce_indices] = 0

    return credit


def select_point(loca_pool_t0, threshold, j, images, point_pos, point_ind, num_global, select_bin, patch_size):
    # print(loca_pool_t0.shape)
    credit = torch.ones(loca_pool_t0.shape[0], 1).to(device)
    point_accum = torch.tensor([]).to(device)

    video = images[: select_bin, :, :, :].permute(0, 3, 1, 2)[None].float()
    zeros = torch.zeros(loca_pool_t0.shape[0], 1).to(device)
    maxima_coords = torch.cat((zeros, loca_pool_t0[:, [1, 0]]), dim = 1).float()  #w,h  
    pred_tracks, pred_visibility = cotracker(video, queries=maxima_coords[None])  

    for i in range(video.shape[1]):
        print('Frame: ', i)
        if i == 0:
            loca = loca_pool_t0
        else:
            loca = pred_tracks[0, i, :, :][:, [1, 0]]  #h,w
        
        vis = pred_visibility[0, i, :] 
        point_accum = torch.cat((point_accum, loca.unsqueeze(0)), dim = 0)
        inter = credit.clone()
        credit[vis == 0] = 0
        
        points_live = point_pos[i, :, :]
        points_live = points_live[~torch.all(points_live == torch.tensor([-1, -1], device='cuda'), dim=1)]
        credit = less_than_one_per_patch(credit, loca, video, patch_size, i, points_live)
        
        print(torch.count_nonzero(credit))

        if torch.count_nonzero(credit) < threshold <= torch.count_nonzero(inter):
            index_all, where_new, where_old = assign_index(credit, inter, threshold)
            where_all = torch.cat((where_old, where_new), dim = 0)

            for o in range(i):
                point_pos[o] = point_accum[o, where_all, :]
                point_ind[o] = index_all[where_all]
                assert (point_pos[o] != -1).all()
                assert (point_ind[o] != -1).all()

            point_pos[i, :torch.count_nonzero(credit)] = point_accum[i, where_new, :]
            point_ind[i, :torch.count_nonzero(credit)] = index_all[where_new]
            num_global += threshold

        if torch.count_nonzero(credit) < threshold and torch.count_nonzero(inter) < threshold:
            where_v3, _ = torch.where(credit == 1.0)
            point_pos[i, :torch.count_nonzero(credit)] = point_accum[i, where_v3, :]
            point_ind[i, :torch.count_nonzero(credit)] = index_all[where_v3]

        if torch.count_nonzero(credit) == 0:
            print('select finished')
            return num_global

    #track till end
    if torch.count_nonzero(credit) > 0:
        new_start = select_bin - 1 ## the last one of before
        while torch.count_nonzero(credit) > 0 and new_start < images.shape[0] - 1:
            
            exist_point = point_pos[new_start, :torch.count_nonzero(credit)]
            exist_index = point_ind[new_start, :torch.count_nonzero(credit)]
            
            point_accum = torch.tensor([]).to(device)
            credit = torch.ones(exist_point.shape[0], 1).to(device)
            
            if new_start + select_bin <= images.shape[0]:
                new_end = new_start + select_bin   
            else:
                new_end = images.shape[0]
            
            print('new_start:', new_start, '&', 'new_end:', new_end, '&', 'new video shape:', video.shape[1])
            
            video = images[new_start: new_end, :, :, :].permute(0, 3, 1, 2)[None].float()
            zeros = torch.zeros(exist_point.shape[0], 1).to(device)
            maxima_coords = torch.cat((zeros, exist_point[:, [1, 0]]), dim = 1).float()  #w,h  
            pred_tracks, pred_visibility = cotracker(video, queries=maxima_coords[None])  
            
            for i in range(video.shape[1]):
                loca = pred_tracks[0, i, :, :][:, [1, 0]]  #h,w
                vis = pred_visibility[0, i, :] 
                point_accum = torch.cat((point_accum, loca.unsqueeze(0)), dim = 0)
                inter = credit.clone()
                credit[vis == 0] = 0

                #1 PER PATCH
                if i > 0:
                    points_live = point_pos[new_start + i, :, :]
                    points_live = points_live[~torch.all(points_live == torch.tensor([-1, -1], device='cuda'), dim=1)]
                    credit = less_than_one_per_patch(credit, loca, video, patch_size, i, points_live)
                #1 PER PATCH END

                if torch.count_nonzero(credit) < threshold and torch.count_nonzero(inter) < threshold:
                    where_v4, _ = torch.where(credit == 1.0)
                    point_pos[new_start + i, :torch.count_nonzero(credit)] = point_accum[i, where_v4, :]
                    point_ind[new_start + i, :torch.count_nonzero(credit)] = exist_index[where_v4]
                    # print(aaa)
                if torch.count_nonzero(credit) == 0:
                    print('select finished')
                    return num_global   
                
            new_start = new_end - 1


    print('--------- select finished --------')
    return num_global



def add_point(loca_pool_t0, num_add, j, images, point_pos, point_ind, num_global, add_bin, patch_size):
    # print(loca_pool_t0.shape, num_add)

    credit = torch.ones(loca_pool_t0.shape[0], 1).to(device)
    point_accum = torch.tensor([]).to(device)
    sums = []
    if j + add_bin <= images.shape[0]:
        video = images[j: j + add_bin, :, :, :].permute(0, 3, 1, 2)[None].float()
        zeros = torch.zeros(loca_pool_t0.shape[0], 1).to(device)
        maxima_coords = torch.cat((zeros, loca_pool_t0[:, [1, 0]]), dim = 1).float()  #w,h  
    else:
        video = images[images.shape[0] - add_bin:, :, :, :].permute(0, 3, 1, 2)[None].float()
        index = torch.ones(loca_pool_t0.shape[0], 1).to(device)
        index = index * (add_bin - (images.shape[0] - j))
        maxima_coords = torch.cat((index, loca_pool_t0[:, [1, 0]]), dim = 1).float()  #w,h

    pred_tracks, pred_visibility = cotracker(video, queries=maxima_coords[None]) 

    if j + add_bin > images.shape[0]:
        video = images[j:, :, :, :].permute(0, 3, 1, 2)[None].float()
        pred_tracks = pred_tracks[:, add_bin - (images.shape[0] - j):, :, :]
        pred_visibility = pred_visibility[:, add_bin - (images.shape[0] - j):, :] 

        assert video.shape[1] == pred_tracks.shape[1]

    for i in range(video.shape[1]):
        print('Frame: ', j + i)
        if i == 0:
            loca = loca_pool_t0
        else:
            loca = pred_tracks[0, i, :, :][:, [1, 0]]  #h,w
            
        vis = pred_visibility[0, i, :] 
        point_accum = torch.cat((point_accum, loca.unsqueeze(0)), dim = 0)
        inter = credit.clone()
        credit[vis == 0] = 0
        
        #1 PER PATCH    
        points_live = point_pos[j + i, :, :]
        points_live = points_live[~torch.all(points_live == torch.tensor([-1, -1], device = 'cuda'), dim=1)]
        credit = less_than_one_per_patch(credit, loca, video, patch_size, i, points_live)
        #1 PER PATCH END
            
        print(torch.count_nonzero(credit))
        
        if torch.count_nonzero(credit) < num_add:
            where_add, _ = torch.where(inter == 1.0)
            
            potential = point_accum[0, where_add]
            existing_where = torch.where(point_ind[j] != -1)[0]
            existing_point = point_pos[j, existing_where]
            for candi in potential:
                distance_min = torch.sqrt(torch.sum((existing_point - candi) ** 2, dim=1)).min()
                sums.append(distance_min)
            
            sums = torch.tensor(sums, device = 'cuda')
            print('the largest distance:', sums.max())
            indice = torch.argsort(sums)[(potential.shape[0] - num_add):]
            assert indice.shape[0] == num_add
            where_add = where_add[indice]
            index_add = torch.arange(num_global, num_global + num_add)

            for g in range(j, j + i):
                start_ind = (point_ind[g] == -1).nonzero(as_tuple=True)[0][0].item()
                point_pos[g, start_ind: start_ind + num_add] = point_accum[g - j, where_add, :]
                point_ind[g, start_ind: start_ind + num_add] = index_add
                # print(aaa)
            num_global += num_add
            
            # print(aa)
            return num_global
    
    
    if torch.count_nonzero(credit) > 0:
        new_start = select_bin - 1 ## the last one of before
        while torch.count_nonzero(credit) > 0 and new_start < images.shape[0] - 1:
            
            exist_point = point_pos[new_start, :torch.count_nonzero(credit)]
            exist_index = point_ind[new_start, :torch.count_nonzero(credit)]
            
            point_accum = torch.tensor([]).to(device)
            credit = torch.ones(exist_point.shape[0], 1).to(device)
            
            if new_start + select_bin <= images.shape[0]:
                new_end = new_start + select_bin   
            else:
                new_end = images.shape[0]
            
            print('new_start:', new_start, '&', 'new_end:', new_end, '&', 'new video shape:', video.shape[1])
            
            video = images[new_start: new_end, :, :, :].permute(0, 3, 1, 2)[None].float()
            zeros = torch.zeros(exist_point.shape[0], 1).to(device)
            maxima_coords = torch.cat((zeros, exist_point[:, [1, 0]]), dim = 1).float()  #w,h  
            pred_tracks, pred_visibility = cotracker(video, queries=maxima_coords[None])  
            
            for i in range(video.shape[1]):
                loca = pred_tracks[0, i, :, :][:, [1, 0]]  #h,w
                vis = pred_visibility[0, i, :] 
                point_accum = torch.cat((point_accum, loca.unsqueeze(0)), dim = 0)
                inter = credit.clone()
                credit[vis == 0] = 0

                #1 PER PATCH
                if i > 0:
                    points_live = point_pos[new_start + i, :, :]
                    points_live = points_live[~torch.all(points_live == torch.tensor([-1, -1], device='cuda'), dim=1)]
                    credit = less_than_one_per_patch(credit, loca, video, patch_size, i, points_live)
                #1 PER PATCH END

                if torch.count_nonzero(credit) < threshold and torch.count_nonzero(inter) < threshold:
                    where_v4, _ = torch.where(credit == 1.0)
                    point_pos[new_start + i, :torch.count_nonzero(credit)] = point_accum[i, where_v4, :]
                    point_ind[new_start + i, :torch.count_nonzero(credit)] = exist_index[where_v4]
                    
                if torch.count_nonzero(credit) == 0:
                    print('select finished')
                    return num_global   
                
            new_start = new_end - 1

    print('----------')
    
    
    # print(aaa)
    where_add, _ = torch.where(credit == 1.0) 
    potential = point_accum[0, where_add]
    existing_where = torch.where(point_ind[j] != -1)[0]
    existing_point = point_pos[j, existing_where]
    for candi in potential:
        distance_min = torch.sqrt(torch.sum((existing_point - candi) ** 2, dim=1)).min()
        sums.append(distance_min)
    
    sums = torch.tensor(sums, device = 'cuda')
    print('the largest distance:', sums.max())
    indice = torch.argsort(sums)[(potential.shape[0] - num_add):]
    assert indice.shape[0] == num_add
    where_add = where_add[indice]
    index_add = torch.arange(num_global, num_global + num_add)

    for g in range(j, j + video.shape[1]):
        start_ind = (point_ind[g] == -1).nonzero(as_tuple=True)[0][0].item()
        point_pos[g, start_ind: start_ind + num_add] = point_accum[g - j, where_add, :]
        point_ind[g, start_ind: start_ind + num_add] = index_add

    num_global += num_add

    return num_global

start_time = time.time()

for i in range(point_pos.shape[0]):
    
    if i == 0:
        loca_pool_t0 = location_pool(images[i], device, patch_size)  #h,w
        num_global = select_point(loca_pool_t0, threshold, i, images, point_pos, point_ind, num_global, select_bin, patch_size)

    else:
        if torch.any(point_ind[i] == -1):
            num_add = torch.sum(point_ind[i] == -1).item()
            loca_pool_t0 = location_pool(images[i], device, patch_size)  #h,w
            num_global = add_point(loca_pool_t0, num_add, i, images, point_pos, point_ind, num_global, add_bin, patch_size)
            print('num_global:',num_global)
            assert torch.any(point_ind[i] == -1) == False
        else:
            print(i, 'pass')

assert torch.any(point_ind == -1) == False
assert torch.any(point_pos == -1) == False

end_time = time.time()
total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)
print(f"Total time taken: {int(minutes)} minutes and {seconds:.2f} seconds")

for i in range(point_pos.shape[0]):
    np.save(save_path + f'/point{i}.npy', point_pos[i].cpu().numpy())
    np.save(save_path + f'/index{i}.npy', point_ind[i].cpu().numpy())
print('num_global:', num_global)
np.save(save_path + f'/num_global.npy', np.array([num_global]))


