import os
import os.path as osp
from torch.utils.data.dataloader import DataLoader
import torch
from cacti.utils.utils import save_image
from cacti.utils.metrics import compare_psnr, compare_ssim
import numpy as np
import einops


def eval_stage_psnr_ssim(stage_outputs):
    psnr_list = [[[0] * 6 for _ in range(4)] for _ in range(8)]  # 4 stages, 6 channels (kk)
    ssim_list = [[[0] * 6 for _ in range(4)] for _ in range(8)]  # 4 stages, 6 channels (kk)
    for jj in range(8):
        for ii in range(4):
            for kk in range(6):
                last_stage = stage_outputs[ii]
                now_stage = stage_outputs[ii + 1]
                last_stage = last_stage[kk][0][0, jj, :, :].detach().cpu().numpy()
                now_stage = now_stage[kk][0][0, jj, :, :].detach().cpu().numpy()
                psnr_list[jj][ii][kk] += compare_psnr(now_stage * 255, last_stage * 255)
                ssim_list[jj][ii][kk] += compare_ssim(now_stage * 255, last_stage * 255)
    stage_psnr_list = np.sum(psnr_list,axis=1)/4
    stage_ssim_list = np.sum(ssim_list,axis=1)/4
    moudle_psnr_list = np.sum(psnr_list,axis=2)/6
    moudle_ssim_list = np.sum(ssim_list,axis=2)/6

    return psnr_list, ssim_list
