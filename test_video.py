import json
import os
import os.path as osp
import sys
import random

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch

from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks
from cacti.utils.utils import save_single_image, get_device_info, load_checkpoints
from cacti.utils.metrics import compare_psnr, compare_ssim
from cacti.utils.config import Config
from cacti.models.builder import build_model
from cacti.datasets.builder import build_dataset
from cacti.utils.logger import Logger
from torch.cuda.amp import autocast
import numpy as np
import argparse
import time
import einops

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

## Model Config
parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging")
# parser.add_argument('--data_path', default='./CAVE_1024_28/', type=str, help='Path of data')
# parser.add_argument('--mask_path', default='./mask_256_28.mat', type=str, help='Path of mask')
parser.add_argument("--size", default=256, type=int, help='The training image size')
parser.add_argument("--reuse", default=[1, 1, 0, 0, 0, 0, 0, 0, 1], type=int, nargs='*', help='Reuse')
parser.add_argument("--stage", default=9, type=str, help='Model scale')
# parser.add_argument("--trainset_num", default=5000, type=int, help='The number of training samples of each epoch')
# parser.add_argument("--testset_num", default=5, type=int, help='Total number of testset')
# parser.add_argument("--seed", default=42, type=int, help='Random_seed')
parser.add_argument("--dim", default=16, type=int, help='The number of channels of Datasets')
parser.add_argument("--batch_size", default=1, type=int, help='Batch_size')
# parser.add_argument("--isTrain", default=False, type=bool, help='Train or test')
parser.add_argument("--bands", default=8, type=int, help='The number of channels of Datasets')
parser.add_argument("--seed", default=42, type=int, help='Random_seed')

# for video_sci
parser.add_argument("config", type=str)
parser.add_argument("--work_dir", type=str)
parser.add_argument("--weights",
                    default='/home/yychen/zhangmuyuan/DPU/work_dirs/DPU_base/log/1000/base/epoch_290.pth',
                    type=str)
parser.add_argument("--device", type=str, default="1")
parser.add_argument("--distributed", type=bool, default=False)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--local_rank", default=0)
parser.add_argument("--body_share_params", default=True)
args = parser.parse_args()
args.device = "cuda" if torch.cuda.is_available() else "cpu"
opt = parser.parse_args()
cfg = Config.fromfile(args.config)
def skip_unserializable_objects(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)  # 可以选择返回字符串描述，也可以返回 None 或其他内容、、

if __name__ == "__main__":
    # args = parse_args()
    # cfg = Config.fromfile(args.config)

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = args.device
    config_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs', config_name)
    mask, mask_s = generate_masks(cfg.test_data.mask_path, cfg.test_data.mask_shape)
    cr = mask.shape[0]
    if args.weights is None:
        args.weights = cfg.checkpoints

    test_dir = osp.join(args.work_dir, "test_images")

    log_dir = osp.join(args.work_dir, "test_log")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k, v) for k, v in device_info.items()])
    logger.info('GPU info:\n'
                + dash_line +
                env_info + '\n' +
                dash_line)
    cfg_copy = cfg.copy()  # 复制一份 cfg，避免对原数据进行修改
    cfg_copy['opt'] = "Skipped opt"  # 或者直接将 opt 排除
    logger.info('cfg info:\n'
                + dash_line +
                json.dumps(cfg_copy, default=skip_unserializable_objects, indent=4) + '\n' +
                dash_line)
    test_data = build_dataset(cfg.test_data, {"mask": mask})
    data_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = build_model(cfg.model).to(device)
    logger.info("Load pre_train model...")
    resume_dict = torch.load(args.weights)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model, model_state_dict, strict=True)

    # print("\nModel's weight statistics:")
    # for name, param in model.named_parameters():
    #     if 'weight' in name:  # 查看权重（通常权重包含在参数名中）
    #         print(f"{name}: Mean = {param.data.mean():.4f}, Std = {param.data.std():.4f}")

    Phi = einops.repeat(mask, 'cr h w->b cr h w', b=1)
    Phi_s = einops.repeat(mask_s, 'h w->b 1 h w', b=1)
    Phi = torch.from_numpy(Phi).to(args.device)
    Phi_s = torch.from_numpy(Phi_s).to(args.device)

    if "partition" in cfg.test_data.keys():
        partition = cfg.test_data.partition
        _, _, Phi_h, Phi_w = Phi.shape
        part_h = partition.height
        part_w = partition.width
        assert (Phi_h % part_h == 0) and (Phi_w % part_w == 0), "Image cannot be chunked!"
        h_num = Phi_h // part_h
        w_num = Phi_w // part_w
        A_Phi = einops.rearrange(Phi, "b cr (h_num h) (w_num w)->(b h_num w_num) cr h w", h=part_h, w=part_w)
        A_Phi_s = einops.rearrange(Phi_s, "b cr (h_num h) (w_num w)->(b h_num w_num) cr h w", h=part_h, w=part_w)

    psnr_dict, ssim_dict = {}, {}
    psnr_list, ssim_list = [], []
    sum_time = 0.0
    time_count = 0

    for data_iter, data in enumerate(data_loader):
        psnr, ssim = 0, 0
        batch_output = []
        meas, gt = data
        gt = gt[0].numpy()
        meas = meas[0].float().to(device)
        batch_size = meas.shape[0]
        name = test_data.data_name_list[data_iter]
        if "_" in name:
            _name, _ = name.split("_")
        else:
            _name, _ = name.split(".")
        out_list = []
        gt_list = []
        for ii in range(batch_size):
            single_gt = gt[ii]
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            if "partition" in cfg.test_data.keys():
                Phi = A_Phi[ii % (h_num * w_num)].unsqueeze(0)
                Phi_s = A_Phi_s[ii % (h_num * w_num)].unsqueeze(0)
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                if "amp" in cfg.keys() and cfg.amp:
                    with autocast():
                        outputs = model(single_meas, Phi, Phi_s)
                else:
                    outputs, nets = model(single_meas, (Phi, Phi_s))
                    # outputs = model(single_meas, (Phi, Phi_s))
                torch.cuda.synchronize()
                end = time.time()
                run_time = end - start
                if ii > 0:
                    sum_time += run_time
                    time_count += 1
            if not isinstance(outputs, list):
                outputs = [outputs]
            output = outputs[-3][0].cpu().numpy().astype(np.float32)
            # output = nets[-2][0].cpu().numpy().astype(np.float32)
            # output = outputs[1][0].cpu().numpy().astype(np.float32)

            if "partition" in cfg.test_data.keys():
                out_list.append(output)
                gt_list.append(single_gt)
                if (ii + 1) % (h_num * w_num) == 0:
                    output = np.array(out_list)
                    single_gt = np.array(gt_list)
                    output = einops.rearrange(output, "(h_num w_num) c cr h w->c cr (h_num h) (w_num w)", h_num=h_num,
                                              w_num=w_num)
                    single_gt = einops.rearrange(single_gt, "(h_num w_num) cr h w->cr (h_num h) (w_num w)", h_num=h_num,
                                                 w_num=w_num)
                    batch_output.append(output)
                    out_list = []
                    gt_list = []
                else:
                    continue
            else:
                batch_output.append(output)
            for jj in range(cr):
                if output.shape[0] == 3:
                    per_frame_out = output[:, jj]
                    rgb2raw = test_data.rgb2raw
                    per_frame_out = np.sum(per_frame_out * rgb2raw, axis=0)
                else:
                    per_frame_out = output[jj]
                per_frame_gt = single_gt[jj]
                psnr += compare_psnr(per_frame_gt * 255, per_frame_out * 255)
                ssim += compare_ssim(per_frame_gt * 255, per_frame_out * 255)
        meas_num = len(batch_output)
        psnr = psnr / (meas_num * cr)
        ssim = ssim / (meas_num * cr)
        logger.info("{}, Mean PSNR: {:.4f} Mean SSIM: {:.4f}.".format(
            _name, psnr, ssim))
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        psnr_dict[_name] = psnr
        ssim_dict[_name] = ssim

        # save image
        out = np.array(batch_output)
        for j in range(out.shape[0]):
            image_dir = osp.join(test_dir, _name)
            if not osp.exists(image_dir):
                os.makedirs(image_dir)
            save_single_image(out[j], image_dir, j, name=config_name)
    if time_count == 0:
        time_count = 1
    logger.info('Average Run Time:\n'
                + dash_line +
                "{:.4f} s.".format(sum_time / time_count) + '\n' +
                dash_line)

    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)

    psnr_str = ", ".join([key + ": " + "{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
    ssim_str = ", ".join([key + ": " + "{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
    logger.info("Mean PSNR: \n" +
                dash_line +
                "{}.\n".format(psnr_str) +
                dash_line)

    logger.info("Mean SSIM: \n" +
                dash_line +
                "{}.\n".format(ssim_str) +
                dash_line)
