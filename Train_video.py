import json
from py3nvml import py3nvml
import time

import torch.utils.data as tud
from einops import einops
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
import os
import torch.distributed as dist
import os.path as osp
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from Utils import *
from Model_DPU import Net_DPU
from Dataset import dataset
from cacti.datasets.builder import build_dataset
from cacti.models.builder import build_model
from cacti.utils import logger
from cacti.utils.config import Config
from cacti.utils.eval import eval_psnr_ssim
from cacti.utils.logger import Logger
from cacti.utils.loss_builder import build_loss
from cacti.utils.mask import generate_masks
from cacti.utils.optim_builder import build_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from cacti.utils.utils import load_checkpoints, save_image, get_device_info

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
## Model Config
parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging")
# parser.add_argument('--data_path', default='./CAVE_1024_28/', type=str, help='Path of data')
# parser.add_argument('--mask_path', default='./mask_256_28.mat', type=str, help='Path of mask')
parser.add_argument("--size", default=128, type=int, help='The training image size')
parser.add_argument("--stage", default=9, type=str, help='Model scale')
# parser.add_argument("--trainset_num", default=5000, type=int, help='The number of training samples of each epoch')
# parser.add_argument("--testset_num", default=5, type=int, help='Total number of testset')
parser.add_argument("--seed", default=42, type=int, help='Random_seed')
# parser.add_argument("--batch_size", default=4, type=int, help='Batch_size')
# parser.add_argument("--isTrain", default=True, type=bool, help='Train or test')
# parser.add_argument("--reuse", default=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], type=int, nargs='*', help='Reuse')
parser.add_argument("--reuse", default=[1, 1, 0, 0, 0, 0, 0, 0, 1], type=int, nargs='*', help='Reuse')
# parser.add_argument("--reuse", default=[1, 0, 0, 0, 1], type=int, nargs='*', help='Reuse')
parser.add_argument("--bands", default=8, type=int, help='The number of channels of Datasets')
parser.add_argument("--dim", default=16, type=int, help='The number of channels of Datasets')
parser.add_argument("--color_dim", default=1, type=int, help='color_ch')
parser.add_argument("--is_train", default=True, type=bool, help='training pattern')
# parser.add_argument("--scene_num", default=1, type=int, help='The number of scenes of Datasets')
# parser.add_argument("--lr", default=0.0004, type=float, help='learning rate')
# parser.add_argument("--len_shift", default=2, type=int, help=' shift length among bands')
# for video_sci
parser.add_argument("config", type=str)
parser.add_argument("--work_dir", type=str, default=None)
parser.add_argument("--device", type=str, default="1")
parser.add_argument("--distributed", type=bool, default=False)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--local_rank", default=0)
parser.add_argument("--body_share_params", default=False)

args = parser.parse_args()
args.device = "cuda" if torch.cuda.is_available() else "cpu"
local_rank = int(args.local_rank)
if args.distributed:
    args.device = torch.device("cuda", local_rank)
opt = parser.parse_args()
cfg = Config.fromfile(args.config)


def loss_f(loss_func, pred, lbl):
    return torch.sqrt(loss_func(pred, lbl))


def skip_unserializable_objects(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)  # 可以选择返回字符串描述，也可以返回 None 或其他内容


if __name__ == "__main__":

    # 初始化 py3nvml
    py3nvml.nvmlInit()


    # 获取 GPU 资源
    def check_gpu():
        handle = py3nvml.nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))  # 获取第一个 GPU
        memory_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        memory = memory_info.free / (1024 ** 3)
        # print(f"{memory}G")
        if memory > 11:  # 假设显存大于 12GB 时认为充足
            return True
        return False


    # 主程序
    while not check_gpu():
        # print("GPU 显存不足，等待 5 秒钟再试...")
        time.sleep(5)

    # 显卡资源充足，运行代码
    print("GPU 资源充足，开始执行代码。")

    # video_sci
    if args.work_dir is None:
        args.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    log_dir = osp.join(args.work_dir, "log")
    show_dir = osp.join(args.work_dir, "show")
    train_image_save_dir = osp.join(args.work_dir, "train_images")
    checkpoints_dir = osp.join(args.work_dir, "checkpoints")
    checkpoints_dir = checkpoints_dir + f'/{cfg.model.type}'
    # checkpoints_dir = '/root/autodl-tmp/checkpoints'
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(show_dir):
        os.makedirs(show_dir)
    if not osp.exists(train_image_save_dir):
        os.makedirs(train_image_save_dir)
    if not osp.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir=show_dir)

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False
    print(opt)

    rank = 0
    if args.distributed:
        local_rank = int(args.local_rank)
        # local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k, v) for k, v in device_info.items()])

    device = args.device
    model = build_model(cfg.model).to(device)

    if rank == 0:
        for key, value in cfg.items():
            print(f'{key}: {type(value)}')

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
        logger.info('Model info:\n'
                    + dash_line +
                    str(model) + '\n' +
                    dash_line)
    # print('time = %s' % (datetime.datetime.now()))
    ## generate video_sci mask
    mask, mask_s = generate_masks(cfg.train_data.mask_path, cfg.train_data.mask_shape)  # mask_s对mask进行归一化
    # video_sci dataset
    train_data = build_dataset(cfg.train_data, {"mask": mask})
    # video_sci optimizer
    optimizer = build_optimizer(cfg.optimizer, {"params": model.parameters()})

    # video_sci 加载数据

    if args.distributed:
        dist_sampler = DistributedSampler(train_data, shuffle=True)
        train_data_loader = DataLoader(dataset=train_data,
                                       batch_size=cfg.data.samples_per_gpu,
                                       sampler=dist_sampler,
                                       num_workers=cfg.data.workers_per_gpu)
    else:
        train_data_loader = DataLoader(dataset=train_data,
                                       batch_size=cfg.data.samples_per_gpu,
                                       shuffle=True,
                                       num_workers=cfg.data.workers_per_gpu)
    criterion = build_loss(cfg.loss)
    criterion = criterion.to(args.device)

    start_epoch = 0
    if rank == 0:
        if cfg.checkpoints is not None:
            logger.info("Load pre_train model...")
            resume_dict = torch.load(cfg.checkpoints)
            if "model_state_dict" not in resume_dict.keys():
                model_state_dict = resume_dict
            else:
                model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model, model_state_dict)
        else:
            logger.info("No pre_train model")

        if cfg.resume is not None:
            logger.info("Load resume...")
            resume_dict = torch.load(cfg.resume)
            start_epoch = resume_dict["epoch"]
            model_state_dict = resume_dict["model_state_dict"]
            load_checkpoints(model, model_state_dict)

            optim_state_dict = resume_dict["optim_state_dict"]
            optimizer.load_state_dict(optim_state_dict)
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # model.train()
    iter_num = len(train_data_loader)

    mse = torch.nn.MSELoss().cuda()

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    for epoch in range(start_epoch, cfg.runner.max_epochs):
        epoch_loss = 0
        model = model.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            gt, meas = data
            gt = gt.float().to(args.device)
            meas = meas.unsqueeze(1).float().to(args.device)
            batch_size = meas.shape[0]

            Phi = einops.repeat(mask, 'cr h w->b cr h w', b=batch_size)
            Phi_s = einops.repeat(mask_s, 'h w->b 1 h w', b=batch_size)

            Phi = torch.from_numpy(Phi).to(args.device)
            Phi_s = torch.from_numpy(Phi_s).to(args.device)

            optimizer.zero_grad()

            model_out = model(meas, (Phi, Phi_s))

            if not isinstance(model_out, list):
                model_out = [model_out]
            # loss = torch.sqrt(criterion(model_out[-1], gt))
            loss = loss_f(mse, model_out[opt.stage - 1], gt) + 0.7 * loss_f(mse, model_out[opt.stage - 2], gt) + \
                   0.5 * loss_f(mse, model_out[opt.stage - 3], gt) + 0.3 * loss_f(mse, model_out[opt.stage - 4], gt)
                    # + 0.01 * loss_f(mse, model_out[1], gt)
            epoch_loss += loss.item()\

            # for name, param in model.named_parameters():
            #     if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            #         print(f"梯度 {name} 包含非法值！")
            #     if torch.isnan(param).any() or torch.isinf(param).any():
            #         print(f"模型参数 {name} 包含非法值！")
            #     if torch.isnan(loss).any() or torch.isinf(loss).any():
            #         print("损失值包含非法值！")
            loss.backward()


            clip_grad_norm_(model.parameters(), max_norm=0.2)  # 梯度截断操作

            optimizer.step()


            # if epoch % 20 == 0 and epoch != 0:
            #     for param_group in optimizer.param_groups:
            #         optimizer.state_dict()["param_groups"][0]["lr"] *= 0.01  # 每10个epoch，学习率乘以0.1

            if rank == 0 and (iteration % cfg.log_config.interval) == 0:
                # lr = optimizer.state_dict()["param_groups"][0]["lr"]
                lr = optimizer.param_groups[0]['lr']
                iter_len = len(str(iter_num))
                logger.info(
                    "epoch: [{}][{:>{}}/{}], lr: {:.6f}, loss: {:.5f}.".format(epoch, iteration, iter_len, iter_num, lr,
                                                                               loss.item()))
                writer.add_scalar("loss", loss.item(), epoch * len(train_data_loader) + iteration)
            # if rank == 0 and (iteration % cfg.save_image_config.interval) == 0:
            #     sing_out = model_out[-1][0].detach().cpu().numpy()
            #     sing_gt = gt[0].cpu().numpy()
            #     image_name = osp.join(train_image_save_dir, str(epoch) + "_" + str(iteration) + ".png")
            #     save_image(sing_out, sing_gt, image_name)
        end_time = time.time()

        # scheduler.step()

        if rank == 0:
            logger.info("epoch: {}, avg_loss: {:.5f}, time: {:.2f}s.\n".format(epoch, epoch_loss / (iteration + 1),
                                                                               end_time - start_time))

        # if rank == 0 and (epoch % cfg.checkpoint_config.interval) == 0:
        #     if args.distributed:
        #         save_model = model.module
        #     else:
        #         save_model = model
        #     checkpoint_dict = {
        #         "epoch": epoch,
        #         "model_state_dict": save_model.state_dict(),
        #         "optim_state_dict": optimizer.state_dict(),
        #     }
        #     torch.save(checkpoint_dict, osp.join(checkpoints_dir, "epoch_" + str(epoch) + ".pth"))

        mask_test, mask_s_test = generate_masks(cfg.test_data.mask_path, cfg.test_data.mask_shape)
        if cfg.eval.flag:
                test_data = build_dataset(cfg.test_data, {"mask": mask_test})
        if rank == 0 and cfg.eval.flag and epoch % cfg.eval.interval == 0:
            if args.distributed:
                psnr_dict, ssim_dict = eval_psnr_ssim(model.module, test_data, mask_test, mask_s_test, args)
            else:
                psnr_dict, ssim_dict = eval_psnr_ssim(model, test_data, mask_test, mask_s_test, args)

            psnr_str = ", ".join([key + ": " + "{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
            ssim_str = ", ".join([key + ": " + "{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
            logger.info("Mean PSNR: \n{}.\n".format(psnr_str))
            logger.info("Mean SSIM: \n{}.\n".format(ssim_str))
