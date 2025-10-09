import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
#from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np

class Dataset(Dataset):
    def __init__(self, args, path, split,
                 width, height, matching_name = 'nearest_1to1', lines_name = 'outside', keep_dim_aspect_ratio=True, preload=True, use_resize_cache=True):
        self.dataset_dir = os.path.dirname(path)
        self.split = split
        self.width = width
        self.height = height
        self.keep_dim_aspect_ratio = keep_dim_aspect_ratio
        self.use_resize_cache = use_resize_cache

        self.dataset_root = os.path.dirname(path)
        self.rgbs_dir = 'extracted_rgb'
        self.poses_dir = 'matched_poses'
        self.lines_dir = 'obj_lines'
        self.matching_name = matching_name
        self.lines_name = lines_name

        self.preload = preload
        #self.noise_sigma = noise_sigma
        #self.t_sigma = t_sigma
        #self.random_rot = random_rot

        #self.cutout_prob = cutout_prob
        #self.use_cutout = cutout_prob > 0.0
        #self.cutout_inside = cutout_inside
        #self.max_cutout_size = max_cutout_size
        #self.min_cutout_size = min_cutout_size

        self.used_size = None

        #if self.split != 'train' and self.cutout_prob > 0.0:
        #    print("***** Split is not train, but cutout is enabled! *****")

        print("Loading dataset from path: ", path)
        with open(path, 'r') as f:
            self.entries = json.load(f)[split]

        # convert paths to host format
        #for i in range(len(self.entries)):
        #    for p in {'exr_normals_path', 'exr_positions_path', 'txt_path'}:
        #        self.entries[i][p] = os.path.join(*self.entries[i][p].split('\\'))

        print("Loading annotations")
        for i in range(len(self.entries)):
            # load lines
            self.entries[i]['lines'] = self.load_lines(self.entries[i])
            self.entries[i]['transform'] = self.load_transform(self.entries[i])
            # add synthetic flag
            #self.entries[i]['synthetic'] = self.is_synthetic(self.entries[i])
            # add ids
            self.entries[i]['sample_id'] = i

        #if 'train' not in path and 'val' not in path:
        #    if self.split == 'train':
        #        self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 != 0]
        #    elif self.split == 'val':
        #        self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 == 0]

        #if args.bins_pick_samples != 'all':
        #    picked = []
        #    print('picking', args.bins_pick_samples, 'samples')
        #    for i in range(len(self.entries)):
        #        if self.entries[i]['synthetic'] is True and args.bins_pick_samples == 'synthetic':
        #            picked.append(self.entries[i])
        #        elif self.entries[i]['synthetic'] is False and args.bins_pick_samples == 'real':
        #            picked.append(self.entries[i])
        #    self.entries = picked
        #    print('Picked', len(self.entries), 'samples')

        if args.mtevents_rgb_subsample_batch < 1.0:
            reduced = []
            for i in range(len(self.entries)):
                if np.random.rand() < args.bins_subsample_batch:
                    reduced.append(self.entries[i])
            #reduced = [self.entries[0], self.entries[1]]
            #reduced = [self.entries[0]]
            self.entries = reduced

        print("Split: ", self.split)
        print("Size: ", len(self))
        if self.preload:
            print("Preloading exrs to memory")
            for entry in self.entries:
                #print(entry)
                entry['rgb'] = self.load_rgb(entry)


        self.means = [125.834, 134.014, 134.206]
        self.stds = [67.9709, 66.5856, 62.999]

    def get_nomalization_constants(self):
        return self.means, self.stds

    def __len__(self):
        """
        Length of dataset
        :return: number of elements in dataset
        """
        return len(self.entries)

    def load_entry_annotation(self, entry):
        sample_name = entry['img_path'].split('.')[0]
        annotation_path = os.path.join(self.dataset_root, self.poses_dir, self.matching_name, sample_name + '.json')
        with open(annotation_path) as f:
            return json.load(f)

    def load_transform(self, entry):
        sample_name = entry['img_path'].split('.')[0]
        annotation_path = os.path.join(self.dataset_root, self.poses_dir, self.matching_name, sample_name + '.json')
        with open(annotation_path) as f:
            annotation = json.load(f)

        obj = None
        for o in annotation['objects']:
            if o['object_id'] == entry['object_id']:
                obj = o
        if obj is None:
            raise(Exception('Object not found int anntation for image', entry['img_path']))

        eangles = obj['euler_angles_deg']
        roll, pitch, yaw = np.deg2rad(eangles[0]), np.deg2rad(eangles[1]), np.deg2rad(eangles[2])

        cx, cy, cz = np.cos([roll, pitch, yaw])
        sx, sy, sz = np.sin([roll, pitch, yaw])

        R_x = np.array([[1, 0, 0],
                        [0, cx, -sx],
                        [0, sx, cx]])

        R_y = np.array([[cy, 0, sy],
                        [0, 1, 0],
                        [-sy, 0, cy]])

        R_z = np.array([[cz, -sz, 0],
                        [sz, cz, 0],
                        [0, 0, 1]])
        R = R_z @ R_y @ R_x

        t = np.array(obj['center_3d'])

        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        return transform

    def load_lines(self, entry):
        lines_file = os.path.join(self.dataset_root, self.lines_dir, self.lines_name,  f'obj_{entry['object_id']:06}.txt')
        #print('lines_file', lines_file)
        #exit()
        lines = np.loadtxt(lines_file)
        return lines

    def get_transformed_lines(self, lines, transform):
        #lines = entry['lines']

        starts = np.c_[(lines[:, :3], np.ones(lines.shape[0]).T)]
        ends = np.c_[(lines[:, 3:], np.ones(lines.shape[0]).T)]

        t_starts = transform @ starts.T
        t_ends = transform @ ends.T

        return np.hstack((t_starts.T[:, :3], t_ends.T[:, :3]))

    def get_resized_size(self, orig_width, orig_height):
        width = self.width
        height = self.height
        if self.keep_dim_aspect_ratio:
            if orig_width > orig_height:
                width = self.width
                height = int(orig_height * (self.width / orig_width))
            else:
                width = int(orig_width * (self.height / orig_height))
                height = self.height
        return width, height



    def load_rgb(self, entry):
        """
        Loads rgb matrix for a given entry
        :param entry: entry from self.entries
        :return: rgb matrix with shape (3, height, width)
        """
        img_path = None
        rgb = None
        cached = False
        if self.use_resize_cache:
            img_path = os.path.join(self.dataset_root, f'_cache_{self.width}x{self.height}', entry['img_path'])
            if os.path.exists(img_path):
                rgb = cv2.imread(img_path)
                if rgb is None:
                    print(img_path)
                    raise ValueError("Image at path ", img_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                #print('Loaded resized sample from cache!')
                cached = True

        if rgb is None:
            #print('reading original')
            img_path = os.path.join(self.dataset_root, self.rgbs_dir, entry['img_path'])
            rgb = cv2.imread(img_path)
            if rgb is None:
                print(img_path)
                raise ValueError("Image at path ", img_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        width, height = self.get_resized_size(rgb.shape[1], rgb.shape[0])
        if self.used_size is not None:
            if width != self.used_size[0] or height != self.used_size[1]:
                raise ValueError("Image at path ", img_path, "has different aspect ratio")
        else:
            self.used_size = (width, height)
            #print(f'Corrected width and height aspect ratio. {rgb.shape[1]} x {rgb.shape[0]} -> {width} x {height}')

        img_resized = False
        if rgb.shape[1] != width or rgb.shape[0] != height:
            rgb = cv2.resize(rgb, (width, height))#, interpolation=cv2.INTER_NEAREST_EXACT)
            img_resized = True

        if self.use_resize_cache and cached is False and img_resized is True:
            img_path = os.path.join(self.dataset_dir, f'_cache_{width}x{height}', entry['img_path'])
            img_dir = os.path.dirname(img_path)
            Path(img_dir).mkdir(parents=True, exist_ok=True)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, rgb)
            #print('Saved resized sample to cache')

        rgb = np.transpose(rgb, [2, 0, 1])
        return rgb

    def get_aug_transform(self):
        """
        Generates random transformation using. R is from SO(3) thanks to QR decomposition.
        :return: random transformation matrix
        """
        if self.random_rot:
            R, _ = np.linalg.qr(np.random.randn(3, 3))
        else:
            R = np.eye(3)

        t = self.t_sigma * np.random.randn(3)

        out = np.zeros([4, 4])
        out[:3, :3] = R
        out[:3, 3] = t
        out[3, 3] = 1

        #if np.random.rand() < 0.5:
        #    out[0, 0] = out[0, 0] * -1

        return out

    def aug(self, xyz_gt, transform):
        """
        Applies transformation matrix to pointcloud
        :param xyz_gt: original pointcloud with shape (3, height, width)
        :param transform: (4, 4) transformation matrix
        :return: Transformed pointcloud with shape (3, height, width)
        """
        orig_shape = xyz_gt.shape
        xyz = np.reshape(xyz_gt, [-1, 3])
        xyz = np.concatenate([xyz, np.ones([xyz.shape[0], 1])], axis=-1)

        xyz_t = (transform @ xyz.T).T

        xyz_t = xyz_t[:, :3] / xyz_t[:, 3, np.newaxis]
        xyz_t = np.reshape(xyz_t, orig_shape)
        return xyz_t

    def normalize_lines(self, lines):
        div = torch.tensor(self.stds + self.stds)
        sub = torch.tensor(self.means + self.means) / div
        #print('sub:', sub)
        #print('div:', div)
        return (lines / div) - sub


    def normalize_rgb(self, rgb):
        r_mean, g_mean, b_mean = self.means
        r_std, g_std, b_std = self.stds

        sub = torch.tensor([r_mean/r_std, g_mean/g_std, b_mean/b_std], dtype=torch.float32).view(3, 1, 1)
        div = torch.tensor([r_std, g_std, b_std], dtype=torch.float32).view(3, 1, 1)
        return (rgb / div) - sub

    def __getitem__(self, index):
        """
        Returns one sample for training
        :param index: index of entry
        :return: dict containing sample data
        """
        entry = self.entries[index]

        transform = np.array(entry['transform'])
        #orig_transform = np.array(entry['orig_transform'])

        #if gt_transform[0, 1] < 0.0:
        #    gt_transform[:, :2] *= -1

        #if self.split == 'train':
        #    aug_transform = self.get_aug_transform()
        #    transform = aug_transform @ gt_transform
        #else:
        #    transform = gt_transform

        transform = transform.astype(np.float32)

        #rot = Rotation.from_matrix(transform[:3, :3])
        #rotvec = torch.from_numpy(rot.as_rotvec())
        #t = torch.from_numpy(transform[:3, 3])

        if self.preload:
            rgb = entry['rgb']
        else:
            rgb = self.load_rgb(entry)

        if self.split == 'train':
            rgb = self.aug(rgb, aug_transform)

        rgb = rgb.astype(np.float32)

        #if self.noise_sigma is not None:
        #    rgb += self.noise_sigma * np.random.randn(*rgb.shape)

        #if self.use_cutout:
        #    if np.random.rand() < self.cutout_prob:
        #        rgb = self.cutout(rgb)

        #visualize_xyz(rgb)

        #return {'rgb': rgb, 'bin_rotvec': rotvec, 'bin_translation': t, 'bin_transform': torch.from_numpy(transform),
        #        'orig_transform': torch.from_numpy(orig_transform), 'txt_path': entry['txt_path']}

        target = {}
        lines = entry['lines']#[[0, 2]]
        target['image_id'] = torch.tensor(entry['sample_id'])
        target['labels'] = torch.tensor([0 for _ in lines], dtype=torch.int64)
        target['area'] = torch.tensor([1 for _ in lines])
        target['iscrowd'] = torch.tensor([0 for _ in lines])
        target['lines'] = torch.tensor(self.get_transformed_lines(lines, transform), dtype=torch.float32)
        #target['exr_file'] = entry['exr_positions_path']

        #print('Lines are', target['lines'])
        #self.normalize_lines(target['lines'])
        #exit()
        #return self.normalize(torch.tensor(rgb)), target

        #rgb = torch.tensor(rgb)

        if True:
            rgb = self.normalize_rgb(torch.tensor(rgb))
            target['lines'] = self.normalize_lines(target['lines'])
            entry['normalized'] = {'means': self.means, 'stds': self.stds}
        else:
            rgb = torch.tensor(rgb)
            entry['normalized'] = {'means': [0.0, 0.0, 0.0], 'stds': [1.0, 1.0, 1.0]}

        #print('target[\'lines\']', target['lines'][:2].view(2, 6))
        #print(target['lines'][:1].shape)
        #exit()
        #target['lines'] = target['lines'][:2].view(2, 6)

        #print('rgb shape', rgb.shape)

        #visualize_xyz_with_lines(rgb.numpy(), target['lines'].numpy())

        #exit()

        #return torch.tensor(self.load_xyz(entry)), target
        return rgb, target, entry

def build_mtevents_rgb(image_set, args):
    if image_set == 'train':
        return Dataset(args, args.mtevents_rgb_dataset_json_path, 'train', args.mtevents_rgb_input_width, args.mtevents_rgb_input_height,
                            preload=not args.mtevents_rgb_no_preload)
    elif image_set == 'val':
        return Dataset(args, args.mtevents_rgb_dataset_json_path, 'val', args.mtevents_rgb_input_width, args.mtevents_rgb_input_height, preload=not args.mtevents_rgb_no_preload)
    elif image_set == 'test':
        return Dataset(args, args.mtevents_rgb_dataset_json_path, 'test', args.mtevents_rgb_input_width, args.mtevents_rgb_input_height, preload=not args.mtevents_rgb_no_preload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='Path to dataset json file.')
    args = parser.parse_args()
    json_path = args.json

    dataset = Dataset(args, json_path, 'val', 800, 600)

    print(dataset[0])

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for item in data_loader:
        #print(item['xyz'].size())
        #xyz = item['xyz'][0].cpu().detach().numpy()

        #print(np.mean(xyz))

        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel(), marker='o')

        #plt.show()
        pass
