import pdb
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import matplotlib.pyplot as plt
import numpy as np

from ..models.lstm.lstm_model import LSTMPredictor
from ..utils import init_logger, set_seed, plot_loss_curve
from ..data_loader.lstm_data_loader import dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='LSTM')
    # Input size
    parser.add_argument('--input_size', default=90, type=int, required=True)
    parser.add_argument('--output_size', default=90, type=int, required=True)
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size of lstm')
    parser.add_argument('--output_length', default=100, type=int, help='output length')
    parser.add_argument('--input_feature', default=12, type=int, required=True)
    parser.add_argument('--output_feature', default=12, type=int, required=True)



    # train
    parser.add_argument('--epochs', default=3, type=int, help='the epochs you want to train')
    parser.add_argument('--loss_type', default='mse', type=str, help='loss type ', required=True)
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int, required=True)
    parser.add_argument('--data_path', default=None, type=str, help='use a space to separate if there is multi datasets', required=True)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    # parser.add_argument('--weight_decay', default=True, type=bool, help='?')
    # others
    parser.add_argument('--log_file', default='', type=str, help='file path of logging', required=True)
    parser.add_argument('--save_path', default='', type=str, help='file path of saving', required=True)
    parser.add_argument('--mode', default='', type=str, help='wanna to train, valid or test the model', choices=['train', 'dev', 'test'], required=True)
    parser.add_argument('--nodes', default=2, type=int, help='the number of gpus used')
    parser.add_argument('--gpu', default=0, type=int, help='single gpu for valid and test')
    parser.add_argument('--figure_path', type=str, help='the path to save figures')
    parser.add_argument('--print_every', default=100, type=int, help='every n iters training logs are generated')
    parser.add_argument('--seed', default=42, type=int, help='seed')


    args = parser.parse_args()
    return args

def train(args, model, experiment):
    if args.output_size == 90:
        log_file = f"{args.log_file}/lstm/short/{args.loss_type}"
    elif args.output_size == 365:
        log_file = f"{args.log_file}/lstm/long/{args.loss_type}"
    else:
        raise ValueError(f"Invalid output size: {args.output_size}")

    if not os.path.exists(log_file):
        os.makedirs(log_file)
    log = init_logger(f"{log_file}/train_{experiment}.log", mode='w')

    # use DDP
    local_rank = int(os.environ["LOCAL_RANK"]) # GPU on current process
    if experiment == 0:
        torch.distributed.init_process_group(backend="nccl") # initialize
    torch.cuda.set_device(local_rank) # bound to local_rank
    print(f'Use GPU: {local_rank} for {args.mode}ing')
    args.nodes = torch.distributed.get_world_size() # total number of gpus
    rank = dist.get_rank() # global rank (0 ~ world_size-1)

    train_dataloader = dataloader(args)
    total_items = len(train_dataloader) * args.batch_size * args.nodes * args.epochs # len(train_dataloader): number of batches in training data, 
    total_steps = total_items // (args.nodes * args.batch_size) # the model will update every step

    if rank == 0:
        log.info(f'{total_items} number of items will be fed into the model.')
        log.info(f'nodes:{args.nodes}, batch_size:{args.batch_size}, epochs:{args.epochs}, total_steps:{total_steps}')
        log.info(' An input example is given as follows:')
        log.info(next(iter(train_dataloader)))

    model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) # Data Parallel

    if args.loss_type.lower() == 'mae':
        criterion = torch.nn.L1Loss().cuda()
    elif args.loss_type.lower() == 'mse':
        criterion = torch.nn.MSELoss().cuda()
    else:
        raise RuntimeError("loss type error. Only accept mae, mse, or huber")
    
    # 优化器设置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999), 
        eps=1e-8,
        weight_decay=1e-4  # 添加权重衰减
    )
    
    # 学习率调度器 - 针对LSTM优化
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 每10个epoch重启一次
        T_mult=2,  # 重启周期翻倍
        eta_min=args.lr * 0.01  # 最小学习率
    )

    model.train()
    start = time.time()
    total_loss = 0
    i = 0
    loss_list = []
    for epoch in tqdm(range(args.epochs)):
        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()
            optimizer.zero_grad()

            outputs = model(inputs, labels)  # 训练时使用teacher forcing
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            total_loss += loss.item()
            optimizer.step()
            i += 1
            # draw(args, outputs.permute(0, 2, 1).cpu().detach().numpy(), torch.cat([inputs, labels], dim=1).permute(0, 2, 1).cpu().detach().numpy())

        loss_avg = total_loss / len(train_dataloader)
        loss_list.append(loss_avg)
        
        # 学习率调度 - 每个epoch更新
        if rank == 0:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            log.info(f"Epoch {epoch + 1}, Learning Rate: {current_lr:.6f}")
            
        if (epoch + 1) % args.print_every == 0:
            step = (i + 1)
            if rank == 0:
                log.info(f'time = {int((time.time() - start) // 60)}m, epoch {epoch + 1}, step = {step}, loss = {loss_avg:.5f},'
                                f'progress:{step*100/total_steps:.1f}%')
            if args.output_size == 90:
                save_path = f"{args.save_path}/lstm/short/{args.loss_type}"
            elif args.output_size == 365:
                save_path = f"{args.save_path}/lstm/long/{args.loss_type}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.module.state_dict(), f"{save_path}/lstm_{experiment}_{epoch + 1}.pth")
        total_loss = 0
        
    plot_loss_curve(loss_list, f'{log_file}/save_loss_curve_{experiment}.png')

def test(args, model, experiment):
    if args.output_size == 90:
        log_file = f"{args.log_file}/lstm/short/{args.loss_type}"
    elif args.output_size == 365:
        log_file = f"{args.log_file}/lstm/long/{args.loss_type}"
    else:
        raise ValueError(f"Invalid output size: {args.output_size}")

    log = init_logger(f"{log_file}/test_{experiment}.log", mode='w')
    device = torch.device("cuda:{}".format(args.gpu))
    print(f'Use GPU: {device} for {args.mode}ing')
    
    test_dataloader = dataloader(args)
    log.info(f'{len(test_dataloader) * args.batch_size} number of items will be tested by models.')
    log.info(f'node_id:{device}, batch_size:{args.batch_size}')

    if args.output_size == 90:
        save_path = f"{args.save_path}/lstm/short/{args.loss_type}"
    elif args.output_size == 365:
        save_path = f"{args.save_path}/lstm/long/{args.loss_type}"

    model.load_state_dict(torch.load(f"{save_path}/lstm_{experiment}_{args.epochs}.pth"), strict=True) # 先DDP，再cuda，再装权重
    model.to(device)
    model.eval()
    log.info(f"Loaded model from {args.save_path}")
    start = time.time()
    if_draw = True

    if args.loss_type.lower() == 'mae':
        criterion = torch.nn.L1Loss().cuda()
    elif args.loss_type.lower() == 'mse':
        criterion = torch.nn.MSELoss().cuda()

    total_loss = 0
    # pdb.set_trace()
    with torch.no_grad():

        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()
            outputs = model(inputs)  # 测试时不使用teacher forcing
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            draw(args, outputs.permute(0, 2, 1).cpu().detach().numpy(), torch.cat([inputs, labels], dim=1).permute(0, 2, 1).cpu().detach().numpy(), experiment)

        avg_loss = total_loss / len(test_dataloader)
        print(f"  预测输出统计:")
        print(f"  输出形状: {outputs.shape}")
        print(f"  输出范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
        print(f"  输出均值: {outputs.mean():.4f}")
        print(f"  输出标准差: {outputs.std():.4f}")
    
    return avg_loss


def draw(args, npList, label, experiment):
    index = random.randint(0, args.batch_size-1)

    prediction_list = npList[index]
    label_list = label[index]

    fig = plt.figure(figsize=(15, 8))
    legend = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3','sub_metering_remainder', 'RR','NBJRR1','NBJRR5' ,'NBJRR10', 'NBJBROU', 'temperature_2m_mean']
    for i in range(14):
        # 创建一个新的子图
        if i >= 9:
            ax = fig.add_subplot(5, 3, i + 1)
        else:
            ax = fig.add_subplot(5, 3, i + 1)

        # 绘制这个特征随时间变化的曲线
        plt.sca(ax)
        x = np.arange(0, args.input_size+args.output_size)

        # import pdb; pdb.set_trace()
        plt.plot(x, label_list[i], label='Ground Truth')
        plt.plot(x[args.input_size:], prediction_list[i], label='Prediction')

        # 添加标签和图例
        plt.title(legend[i])
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value')
        plt.legend()

    # 保存图像
    plt.tight_layout()
    if args.output_size == 90:
        savefig_path = f"{args.figure_path}/lstm/short/{args.loss_type}"
    elif args.output_size == 365:
        savefig_path = f"{args.figure_path}/lstm/long/{args.loss_type}"
    if not os.path.exists(savefig_path):
        os.makedirs(savefig_path)
    plt.savefig(f"{savefig_path}/features_{experiment}.png")

def main():
    args = parse_args()
    set_seed(args)
    all_losses = [] 
    for experiment in range(5):
        print(f"Starting experiment {experiment + 1}/5")
        args.mode = 'train'
        model = LSTMPredictor(args)
        train(args, model, experiment)

        args.mode = 'test'
        model = LSTMPredictor(args)
        avg_loss = test(args, model, experiment)
        all_losses.append(avg_loss)
    print("\nAverage over 5 experiments:")
    print(f"Average Loss: {np.mean(all_losses):.5f} ± {np.std(all_losses):.5f}")
if __name__ == "__main__":
    main()