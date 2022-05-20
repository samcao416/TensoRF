import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T


from .ray_utils import *

class BigSceneDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack #?
        self.downsample = downsample
        #self.img_wh
        self.define_transforms()

        self.scene_bbox = torch.Tensor([[-4.0, -4.0, -4.0,], [4.0, 4.0, 4.0]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if self.split == 'render':
            self.read_meta_render()
        else:
            self.read_meta()
        self.define_proj_mat()

        self.white_bg = False
        self.near_far = [2.0, 6.0]

        self.center = torch.mean(self.scene_bbox, axis = 0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_meta(self):

        with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
            self.meta = json.load(f)

        w = int(self.meta['w'] / self.downsample)
        h = int(self.meta['h'] / self.downsample)
        self.img_wh = [w, h]
        self.focal = self.meta['fl_x'] * (w / self.meta['w'])


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal, self.focal])  # (h, w, 3)
        self.directoins = self.directions / torch.norm(self.directions, dim = -1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depths = []
        #self.downsample = 1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis

        if self.split != 'train' and img_eval_interval == 1:
            img_eval_interval = 5

        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'): #img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses +=[c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS) # Image.LANCZOS is a interpolation algorithm
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1,0) # (h*w, 3) RGB
            self.all_rgbs += [img]

            
            rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)] # (h*w, 6)
        

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames']) * h * w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(slef.meta['frames']) * h * w, 3)

        else:
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3) # (len(self.meta['frames]),h,w,3)

    def read_meta_render(self):

        with open(os.path.join(self.root_dir, "cam_tra", "camera_render.json"), 'r') as f:
            self.meta = json.load(f)
        
        w = int(self.meta['w'] / self.downsample)
        h = int(self.meta['h'] / self.downsample)
        self.img_wh = [w, h]
        self.focal = self.meta['fl_x'] * (w / self.meta['w'])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal, self.focal])  # (h, w, 3)
        self.directoins = self.directions / torch.norm(self.directions, dim = -1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depths = []

        img_eval_interval = 1 if self.N_vis <0 else len(self.meta['frames']) // self.N_vis

        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'): #img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses +=[c2w]

            rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)] # (h*w, 6)
        
        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames']) * h * w, 3)

        else:
            self.all_rays = torch.stack(self.all_rays, 0)

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]
    
    def world2ndc(self, points, lindisp = None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
        
        #elif self.split == 'test':
        #    rays = self.all_rays[idx]
#
        #    sample = {'rays': self.all_rays[idx],
        #    }


        else: # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            #mask = self.all_masks[idx]

            sample = {'rays': rays, 
                      'rgbs': img, 
                     }
                      #'mask': mask}
        return sample

