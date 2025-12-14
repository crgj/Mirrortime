import torch
from torch.utils.data import Dataset
from utils.camera_utils import loadCam

class FourDDataset(Dataset):
    def __init__(self, cam_infos, args, dataset_config):
        self.cam_infos = cam_infos
        self.args = args # ModelParams
        # dataset_config can be e.g., resolution_scale
        self.resolution_scale = dataset_config.get("resolution_scale", 1.0)
        self.is_nerf_synthetic = dataset_config.get("is_nerf_synthetic", False)
        self.is_test_dataset = dataset_config.get("is_test_dataset", False)

    def __len__(self):
        return len(self.cam_infos)

    def __getitem__(self, idx):
        cam_info = self.cam_infos[idx]
        # loadCam arguments: args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset
        cam = loadCam(self.args, idx, cam_info, self.resolution_scale, self.is_nerf_synthetic, self.is_test_dataset)
        return cam
