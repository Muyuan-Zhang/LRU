import numpy as np
import h5py  # 使用 h5py 加载 MATLAB v7.3 文件
from torch.utils.data import Dataset
import os
import os.path as osp
from .builder import DATASETS

@DATASETS.register_module
class middle_scale(Dataset):
    def __init__(self, data_root, *args, **kwargs):
        self.data_root = data_root
        self.data_name_list = os.listdir(data_root)
        self.mask = kwargs["mask"]
        self.frames, self.height, self.width = self.mask.shape

    def __getitem__(self, index):
        file_path = osp.join(self.data_root, self.data_name_list[index])
        with h5py.File(file_path, 'r') as f:
            # 加载 MATLAB v7.3 文件中的数据
            if 'orig_bayer' in f:
                pic = np.array(f['orig_bayer'])
            elif 'patch_save' in f:
                pic = np.array(f['patch_save'])
            elif 'p1' in f:
                pic = np.array(f['p1'])
            elif 'p2' in f:
                pic = np.array(f['p2'])
            elif 'p3' in f:
                pic = np.array(f['p3'])
            else:
                raise KeyError(f"未找到有效的数据字段：{file_path}")

        pic = pic / 255  # 归一化
        pic = pic.transpose(2, 1, 0)
        pic = pic[0:self.height, 0:self.width, :]  # 裁剪到指定大小

        # 处理数据
        pic_gt = np.zeros([pic.shape[2] // self.frames, self.frames, self.height, self.width])
        for jj in range(pic.shape[2]):
            if jj % self.frames == 0:
                meas_t = np.zeros([self.height, self.width])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = self.mask[n, :, :]

            pic_gt[jj // self.frames, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t, pic_t)

            if jj == (self.frames - 1):
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % self.frames == 0 and jj != (self.frames - 1):
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)

        return meas, pic_gt

    def __len__(self):
        return len(self.data_name_list)