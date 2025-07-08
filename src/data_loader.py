import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class PowerDataset(Dataset):
    def __init__(self, args):
        if args.mode == 'train':
            df = pd.read_csv(args.data_path + '/train_new.csv')
        elif args.mode == 'test':
            df = pd.read_csv(args.data_path + '/test_new.csv')

        # 数据清理
        df.replace('?', np.nan, inplace=True)
        df.dropna(inplace=True)
        df = df.drop(columns=['DateTime'])
        
        # 使用MinMaxScaler进行标准化（范围0-1）
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(np.array(df))
        
        print(f"📊 数据形状: {self.data.shape}")
        print(f"📊 数据范围: [{self.data.min():.3f}, {self.data.max():.3f}]")
        print(f"📊 各特征统计:")
        for i, col in enumerate(df.columns):
            print(f"  {col}: [{self.data[:, i].min():.3f}, {self.data[:, i].max():.3f}]")

        self.input_size = args.input_size
        self.output_size = args.output_size

        self.data_x = []
        self.data_yin = []
        self.data_yout = []
        self.split_data(args)

    def split_data(self, args):
        dataX = []  # 保存X
        dataY = []  # 保存Y

        # 将输入窗口的数据保存到X中，将输出窗口保存到Y中
        window_size = self.input_size + self.output_size
        for index in range(len(self.data) - window_size):
            dataX.append(self.data[index: index + self.input_size][:])
            dataY.append(self.data[index + self.input_size: index + window_size][:])
        print(f"📊 生成了 {len(dataX)} 个训练样本")

        self.data_x = np.array(dataX)
        self.data_y = np.array(dataY)
    
    def __len__(self):
        # 返回数据的总数
        return len(self.data_x)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.data_x[idx])
        label = torch.tensor(self.data_y[idx])
        return data, label
    
def dataloader(args):
    raw_dataset = PowerDataset(args)
    if args.mode == 'train':
        sampler = DistributedSampler(raw_dataset)
        dataloader = DataLoader(raw_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    else:
        dataloader = DataLoader(raw_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True) # 单卡

    return dataloader