import os
import random
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt

def set_seed(config):
    seed = config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def init_logger(log_file=None, log_file_level=logging.NOTSET, mode='a'):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def plot_loss_curve(loss_list, save_path):
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)
    plt.close()

def create_masks(src, tgt):
    # src: [bs, n]
    # tgt: [bs, m-1]
    def get_pad_mask(src):
        return (src == 0).unsqueeze(-2).to('cuda')
    src_mask = get_pad_mask(src) # [bs, 1, 90]

    def get_subsequent_mask(tgt): # 上三角矩阵
        bs, len_q = tgt.size()
        subsequent_mask = (torch.triu(torch.ones((1, len_q, len_q)), diagonal=1)).bool()
        # triu(, diagonal=1) 保留主对角线上面一行，及其往上的全部
        return subsequent_mask.to('cuda')
    tgt_mask = get_pad_mask(tgt) | get_subsequent_mask(tgt) # decoder自己本来对句末padding的mask和遮蔽当前时刻后的mask叠加

    return src_mask, tgt_mask

def draw(args, npList, label, experiment):
    index = random.randint(0, args.batch_size-1)

    prediction_list = npList[index]
    label_list = label[index]

    fig = plt.figure(figsize=(15, 8))
    legend = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'sub_metering_remainder', 'RR','NBJRR1','NBJRR5' ,'NBJRR10', 'NBJBROU', 'temperature_2m_mean']
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
        savefig_path = f"{args.figure_path}/transformer/short/{args.loss_type}"
    elif args.output_size == 365:
        savefig_path = f"{args.figure_path}/transformer/long/{args.loss_type}"
    if not os.path.exists(savefig_path):
        os.makedirs(savefig_path)
    plt.savefig(f"{savefig_path}/features_{experiment}.png")