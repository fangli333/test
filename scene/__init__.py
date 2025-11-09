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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        # elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        #     print("Found transforms_train.json file, assuming Blender data set!")
        #     scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
            # dataset_type="blender"
        # elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
        #     scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            # dataset_type="dynerf"

        ##ROS-Cam
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
            dataset_type="nerfies"

        elif 'DAVIS' in args.source_path:
            print('Processing DAVIS data set!')
            scene_info = sceneLoadTypeCallbacks["DAVIS"](args.source_path, args.eval)
            dataset_type = 'davis'

        elif 'MPI-Sintel' in args.source_path:
            print('Processing MPI-Sintel data set!')
            scene_info = sceneLoadTypeCallbacks["MPI-Sintel"](args.source_path, args.eval)
            dataset_type = 'mpi-sintel'
        elif 'iphone' in args.source_path:
            print('Processing iPhone data set!')
            scene_info = sceneLoadTypeCallbacks["iPhone"](args.source_path, args.eval)
            dataset_type = 'iphone'
            
        elif 'tum' in args.source_path:
            print('Processing TUM data set!')
            scene_info = sceneLoadTypeCallbacks["TUM"](args.source_path, args.eval)
            dataset_type = 'tum'

        else:
            assert False, "Could not recognize scene type!"




        # self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_camera = cameraList_from_camInfos(scene_info.train_cameras, resolution_scales, args)
        print('args.addpoints:', args.add_points)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)


    def getTrainCameras(self, scale=1.0):
        return self.train_camera
