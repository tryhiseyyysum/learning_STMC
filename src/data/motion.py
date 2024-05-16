import os
import torch
import numpy as np
import random


class AMASSMotionLoader:
    def __init__(
        self, base_dir, fps, disable: bool = False, nfeats=None, umin_s=0.5, umax_s=3.0
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.disable = disable
        self.nfeats = nfeats

        # unconditional, sampling the duration from [umin, umax]
        self.umin = int(self.fps * umin_s)
        assert self.umin > 0
        self.umax = int(self.fps * umax_s)

    def __call__(self, path, start, end, drop_motion_perc=None, load_transition=False):
        if self.disable:
            return {"x": path, "length": int(self.fps * (end - start))}

        # if path[0]=='M':
        #     path=path[2:]
        # load the motion
        if path not in self.motions:
            motion_path = os.path.join(self.base_dir, path + ".npy")
            # print('-----------------------')
            # print(path)
            # print('-------------------')
            motion = np.load(motion_path)
            motion = torch.from_numpy(motion).to(torch.float)
            self.motions[path] = motion

        if load_transition:
            motion = self.motions[path]
            # take a random crop
            duration = random.randint(self.umin, min(self.umax, len(motion)))
            # random start
            start = random.randint(0, len(motion) - duration)
            motion = motion[start : start + duration]
        else:
            begin = int(start * self.fps)
            end = int(end * self.fps)

            motion = self.motions[path][begin:end]

            # crop max X% of the motion randomly beginning and end
            if drop_motion_perc is not None:
                max_frames_to_drop = int(len(motion) * drop_motion_perc)
                # randomly take a number of frames to drop
                n_frames_to_drop = random.randint(0, max_frames_to_drop)

                # split them between left and right
                n_frames_left = random.randint(0, n_frames_to_drop)
                n_frames_right = n_frames_to_drop - n_frames_left

                # crop the motion
                motion = motion[n_frames_left:-n_frames_right]

        x_dict = {"x": motion, "length": len(motion)}
        return x_dict


class Normalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = base_dir
        base_dir = 'datasets/stats/humanml3d/guoh3dfeats'
        self.mean_path = os.path.join(base_dir, "mean.pt")
        self.std_path = os.path.join(base_dir, "std.pt")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        self.mean = torch.load(self.mean_path)
        self.std = torch.load(self.std_path)

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean, self.mean_path)
        torch.save(std, self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x
