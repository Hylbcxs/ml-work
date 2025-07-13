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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import numpy as np

from ..models.our.mtlfs_model import MTLFS_Model
from ..utils import init_logger, set_seed, plot_loss_curve
from ..data_loader.mtlfs_data_loader import mtlfs_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='MTLFS Model')
    
    # Model Parameters
    parser.add_argument('--input_feature', default=14, type=int, help='input feature size')
    parser.add_argument('--hidden_size', default=256, type=int, help='hidden size')
    parser.add_argument('--input_size', default=90, type=int, help='input sequence length')
    parser.add_argument('--output_size', default=90, type=int, help='short-term prediction length')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    
    # Training Parameters
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--loss_type', default='mse', type=str, help='loss type: mse, mae')
    
    # Validation Parameters
    parser.add_argument('--val_ratio', default=0.2, type=float, help='validation ratio (0.2 = 20%)')
    parser.add_argument('--early_stop_patience', default=20, type=int, help='early stopping patience')
    parser.add_argument('--val_check_interval', default=5, type=int, help='validation check interval (epochs)')
    
    # Data Parameters
    parser.add_argument('--data_path', required=True, type=str, help='data path')
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--prediction_mode', default='short', type=str, 
                       help='prediction mode: short, long')
    
    # Other Parameters
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--print_every', default=10, type=int, help='print frequency')
    parser.add_argument('--log_file', required=True, type=str, help='log file path')
    parser.add_argument('--save_path', required=True, type=str, help='model save path')
    parser.add_argument('--figure_path', required=True, type=str, help='figure save path')
    parser.add_argument('--num_workers', default=4, type=int, help='data loader workers')
    
    return parser.parse_args()


def train(args, model, experiment, prediction_type):
    output_size = args.output_size
    log_file = f"{args.log_file}/mtlfs/{prediction_type}/{args.loss_type}"
    os.makedirs(log_file, exist_ok=True)
    log = init_logger(f"{log_file}/train_{experiment}.log", mode='w')

    # 分布式训练设置
    local_rank = int(os.environ["LOCAL_RANK"])
    if experiment == 0:
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    args.distributed = torch.distributed.is_initialized() if torch.distributed.is_available() else False
    args.nodes = torch.distributed.get_world_size()
    rank = dist.get_rank()

    # 数据加载器 - 使用单一预测模式
    train_loader, val_loader = mtlfs_dataloader(args, prediction_type, args.val_ratio)
    
    if rank == 0:
        log.info(f'MLTFS {prediction_type}期预测训练开始')
        log.info(f'预测长度: {output_size}天')

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
    )

    # 损失函数和优化器
    if args.loss_type.lower() == 'mae':
        criterion = torch.nn.L1Loss().cuda()
    elif args.loss_type.lower() == 'mse':
        criterion = torch.nn.MSELoss().cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10,
        verbose=True if rank == 0 else False
    )

    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    if rank == 0:
        log.info("🏁 开始训练...")
    
    i = 0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(inputs, prediction_type=prediction_type)
            loss = criterion(predictions, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_batches += 1
            if i % 100 == 0:
            
                draw(args, predictions.permute(0, 2, 1).cpu().detach().numpy(), torch.cat([inputs, labels], dim=1).permute(0, 2, 1).cpu().detach().numpy(), experiment)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if (epoch + 1) % args.val_check_interval == 0 or epoch == args.epochs - 1:
            avg_val_loss = validate_model(
                model.module if args.distributed else model, 
                val_loader, 
                criterion, 
                prediction_type
            )
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型
                if rank == 0:
                    save_path = f"{args.save_path}/mtlfs/{prediction_type}/{args.loss_type}"
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(
                        model.module.state_dict() if args.distributed else model.state_dict(), 
                        f"{save_path}/mtlfs_model_{experiment}.pth"
                    )
                    log.info(f"💾 新的最佳模型已保存 (Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f})")
            else:
                patience_counter += 1
            
            # 日志记录
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log.info(f"Epoch {epoch+1}/{args.epochs}")
                log.info(f"  训练损失: {avg_train_loss:.6f}")
                log.info(f"  验证损失: {avg_val_loss:.6f}")
                log.info(f"  最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch+1})")
                log.info(f"  学习率: {current_lr:.8f}")
                log.info(f"  早停计数: {patience_counter}/{args.early_stop_patience}")
            # 早停检查
            if patience_counter >= args.early_stop_patience:
                if rank == 0:
                    log.info(f"早停触发！最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch+1})")
                break
        else:
            # 非验证epoch只记录训练损失
            if rank == 0 and (epoch + 1) % args.print_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                log.info(f"Epoch {epoch+1}/{args.epochs}")
                log.info(f"  训练损失: {avg_train_loss:.6f}")
                log.info(f"  学习率: {current_lr:.8f}")
        # if rank == 0:
        #     scheduler.step()
            
        #     if (epoch + 1) % args.print_every == 0:
        #         log.info(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.5f}')
                
        #         # 保存模型
        #         save_path = f"{args.save_path}/mltfs/{prediction_type}/{args.loss_type}"
        #         if not os.path.exists(save_path):
        #             os.makedirs(save_path)
        #         torch.save(model.module.state_dict(), 
        #                   f"{save_path}/mltfs_{prediction_type}_{experiment}_{epoch + 1}.pth")
    
    plot_tran_and_valid_loss(args, train_losses, val_losses, experiment, best_epoch, log_file)
        
    # plot_loss_curve(loss_list, f'{log_file}/save_loss_curve_{experiment}.png')

def validate_model(model, val_loader, criterion, prediction_type):
    """
    验证模型性能
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        prediction_type: 预测类型
    
    Returns:
        avg_val_loss: 平均验证损失
    """
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()
            
            # 前向传播
            predictions = model(inputs, prediction_type=prediction_type)
            loss = criterion(predictions, labels)
            
            total_val_loss += loss.item()
            num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches
    return avg_val_loss

def plot_tran_and_valid_loss(args, train_losses, val_losses, experiment, best_epoch, log_file):
    plt.figure(figsize=(12, 5))
    
    # 训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='training loss', color='blue')
    plt.title('training loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 验证损失曲线
    if val_losses:
        plt.subplot(1, 2, 2)
        val_epochs = [(i+1) * args.val_check_interval - 1 for i in range(len(val_losses))]
        if val_epochs[-1] != args.epochs - 1:  # 如果最后一个epoch有验证
            val_epochs[-1] = len(train_losses) - 1
        plt.plot(val_epochs, val_losses, label='validation loss', color='red', marker='o')
        plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'best model (Epoch {best_epoch+1})')
        plt.title('validation loss curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    # log_path = f"{args.log_file}/mtlfs_with_val/{prediction_type}/{args.loss_type}"
    # os.makedirs(log_path, exist_ok=True)
    plt.savefig(f"{log_file}/training_curves_{experiment}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
def test(args, model, prediction_type, experiment):
    # 创建测试数据加载器
    test_dataloader = mtlfs_dataloader(args, prediction_type)
    test_model_path = f"{args.save_path}/mltfs/{prediction_type}/{args.loss_type}/mtlfs_best_model_{experiment}.pth"
    # 加载训练好的模型
    if test_model_path and os.path.exists(test_model_path):
        model.load_state_dict(torch.load(test_model_path))
    else:
        print("警告: 未指定模型路径，使用随机初始化的模型")
    
    model = model.cuda()
    model.eval()

    if args.loss_type.lower() == 'mae':
        criterion = torch.nn.L1Loss().cuda()
    elif args.loss_type.lower() == 'mse':
        criterion = torch.nn.MSELoss().cuda()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    print("🚀 开始预测...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_dataloader)):
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()
            
            # 前向传播
            predictions = model(inputs, prediction_type=prediction_type)
            loss = criterion(predictions, labels)

            all_predictions.append(predictions)
            all_targets.append(labels)
            total_loss += loss.item()

            draw(args, predictions.permute(0, 2, 1).cpu().detach().numpy(), torch.cat([inputs, labels], dim=1).permute(0, 2, 1).cpu().detach().numpy(), experiment)
    avg_loss = total_loss / len(test_dataloader)
    
    # 合并所有预测结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return avg_loss
    
def draw(args, npList, label, experiment):
    # index = random.randint(0, args.batch_size-1)
    actual_batch_size = npList.shape[0]
    index = random.randint(0, actual_batch_size - 1)

    prediction_list = npList[index]
    label_list = label[index]

    fig = plt.figure(figsize=(15, 8))
    # legend = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'sub_metering_remainder', 'RR','NBJRR1','NBJRR5' ,'NBJRR10', 'NBJBROU', 'temperature_2m_mean']
    legend = ['Global_active_power']
    for i in range(1):
        # 创建一个新的子图
        # if i >= 9:
        #     ax = fig.add_subplot(5, 3, i + 1)
        # else:
        #     ax = fig.add_subplot(5, 3, i + 1)
        ax = fig.add_subplot(1, 1, 1)

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
        savefig_path = f"{args.figure_path}/mtlfs/{args.prediction_mode}/{args.loss_type}"
    elif args.output_size == 365:
        savefig_path = f"{args.figure_path}/mtlfs/{args.prediction_mode}/{args.loss_type}"
    if not os.path.exists(savefig_path):
        os.makedirs(savefig_path)
    plt.savefig(f"{savefig_path}/features_{experiment}.png")
    plt.close()
    

def main():
    args = parse_args()
    set_seed(args)
    all_losses = [] 

    for experiment in range(5):
        print(f"Starting experiment {experiment + 1}/5")
        args.mode = 'train'
        # 创建模型
        model = MTLFS_Model(
            input_feature=args.input_feature,
            hidden_size=args.hidden_size,
            input_size=args.input_size,
            output_size=args.output_size,
            dropout=args.dropout
        )
        train(args, model, experiment, args.prediction_mode)

        args.mode = 'test'
        model = MTLFS_Model(
            input_feature=args.input_feature,
            hidden_size=args.hidden_size,
            input_size=args.input_size,
            output_size=args.output_size,
            dropout=args.dropout
        )
        avg_loss = test(args, model, args.prediction_mode, experiment)
        all_losses.append(avg_loss)
    print("\nAverage over 5 experiments:")
    print(f"Average Loss: {np.mean(all_losses):.5f} ± {np.std(all_losses):.5f}")


if __name__ == '__main__':
    main() 