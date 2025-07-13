import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import pickle

class MTLFS_PowerDataset(Dataset):

    def __init__(self, args, prediction_type='short'):
        
        self.args = args
        self.input_size = args.input_size
        self.prediction_type = prediction_type
        self.output_size = 90 if prediction_type == 'short' else 365
        self.scaler_path = f"{args.data_path}/scaler.pkl"

        # 加载和预处理数据
        self._load_and_preprocess_data(args)
        
        # 生成样本
        self.data_x = []
        self.data_y = []
        self._prepare_data()
    
    def _load_and_preprocess_data(self,args):
        """加载和预处理数据，确保训练和测试使用相同的标准化参数"""
        if self.args.mode in ['train', 'validation']:
            # 训练模式：加载训练数据，拟合scaler并保存
            print("{self.args.mode}模式：使用训练数据")
            df = pd.read_csv(f"{self.args.data_path}/train_new.csv")
            df = self._clean_data(df)
            if self.args.mode == 'train':
            
                # 拟合scaler
                self.scaler = RobustScaler()
                self.data = self.scaler.fit_transform(np.array(df))
                
                # 保存scaler参数供测试时使用
                os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                print(f"Scaler已保存到: {self.scaler_path}")
            else:
                if os.path.exists(self.scaler_path):
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    self.data = self.scaler.transform(np.array(df))
                    print(f"Scaler已加载用于验证")
                else:
                    raise FileNotFoundError(f"找不到scaler文件: {self.scaler_path}")
            
        elif self.args.mode == 'test':

            df = pd.read_csv(f"{self.args.data_path}/test_new.csv")
            df = self._clean_data(df)
            
            # 加载训练时保存的scaler
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Scaler已加载: {self.scaler_path}")
                
                # 使用相同的scaler转换测试数据（只transform，不fit）
                self.data = self.scaler.transform(np.array(df))
            else:
                raise FileNotFoundError(f"找不到scaler文件: {self.scaler_path}")
                
        # 数据统计（用于验证标准化是否正确）
        print(f"数据形状: {self.data.shape}")
        print(f"数据范围: [{self.data.min():.4f}, {self.data.max():.4f}]")
        print(f"数据均值: {self.data.mean():.4f}")
        print(f"数据标准差: {self.data.std():.4f}")
        
        # 验证：训练和测试数据的分布应该相似
        if hasattr(self, '_train_stats') and self.args.mode == 'test':
            print("与训练数据分布对比:")
            print(f"训练数据范围: {self._train_stats['range']}")
            print(f"测试数据范围: [{self.data.min():.4f}, {self.data.max():.4f}]")
        elif self.args.mode == 'train':
            # 保存训练数据统计用于对比
            self._train_stats = {
                'range': f"[{self.data.min():.4f}, {self.data.max():.4f}]",
                'mean': self.data.mean(),
                'std': self.data.std()
            }
    
    def _clean_data(self, df):
        """数据清理"""
        df.replace('?', np.nan, inplace=True)
        df.dropna(inplace=True)
        df = df.drop(columns=['DateTime'])
        return df

    def _prepare_data(self):
        window_size = self.input_size + self.output_size
        
        for index in range(len(self.data) - window_size):
            input_data = self.data[index:index + self.input_size]
            label_data = self.data[index + self.input_size:index + window_size]
            
            self.data_x.append(input_data)
            self.data_y.append(label_data)
        
        self.data_x = np.array(self.data_x)
        self.data_y = np.array(self.data_y)
        
        print(f"{self.prediction_type}期预测数据准备完成")
        print(f"生成样本数: {len(self.data_x)}")
        print(f"输入形状: {self.data_x.shape}")
        print(f"标签形状: {self.data_y.shape}")
    
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        input_data = torch.tensor(self.data_x[idx], dtype=torch.float32)
        label_data = torch.tensor(self.data_y[idx], dtype=torch.float32)
        
        return input_data, label_data
def create_train_val_split_indices(total_samples, val_ratio=0.2, seed=42):
    """
    创建训练和验证集的索引分割
    
    Args:
        total_samples: 总样本数
        val_ratio: 验证集比例 (0.2表示20%用作验证集)
        seed: 随机种子
    
    Returns:
        train_indices, val_indices: 训练和验证集的索引
    """
    np.random.seed(seed)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    val_size = int(total_samples * val_ratio)
    val_indices = indices[:val_size]
    # val_size = int(total_samples * val_ratio)
    # train_indices = np.arange(total_samples - val_size)
    # val_indices = np.arange(total_samples - val_size, total_samples)
    train_indices = indices[val_size:]
    
    return train_indices, val_indices

def mtlfs_dataloader(args, prediction_type='short', val_ratio=0.2):
    """
    创建包含验证集的MTLFS数据加载器
    
    Args:
        args: 训练参数
        prediction_type: 'short' or 'long'
        val_ratio: 验证集比例 (默认0.2，即20%)
    
    Returns:
        根据args.mode返回相应的数据加载器
        - 如果mode='train'，返回(train_loader, val_loader)
        - 如果mode='test'，返回test_loader
    """
    
    if args.mode == 'train':
        # 创建训练数据集来获取总样本数
        full_dataset = MTLFS_PowerDataset(args, prediction_type)
        total_samples = len(full_dataset)
        
        # 分割训练和验证集索引
        train_indices, val_indices = create_train_val_split_indices(
            total_samples, val_ratio, seed=args.seed if hasattr(args, 'seed') else 42
        )
        
        print(f"数据集分割信息:")
        print(f"   总样本数: {total_samples}")
        print(f"   训练样本数: {len(train_indices)} ({len(train_indices)/total_samples*100:.1f}%)")
        print(f"   验证样本数: {len(val_indices)} ({len(val_indices)/total_samples*100:.1f}%)")
        
        # 创建训练集子集
        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)
        
        # 创建分布式采样器（如果使用分布式训练）
        if hasattr(args, 'distributed') and args.distributed:
            train_sampler = DistributedSampler(train_subset)
            val_sampler = DistributedSampler(val_subset)
            shuffle_train = False
            shuffle_val = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle_train = True
            shuffle_val = False  # 验证集不需要shuffle
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_subset, 
            batch_size=args.batch_size, 
            sampler=train_sampler,
            shuffle=shuffle_train,
            drop_last=True,
            num_workers=getattr(args, 'num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=args.batch_size, 
            sampler=val_sampler,
            shuffle=shuffle_val,
            drop_last=False,  # 验证集保留所有数据
            num_workers=getattr(args, 'num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
        
    elif args.mode == 'test':
        # 测试模式：创建测试数据集
        test_dataset = MTLFS_PowerDataset(args, prediction_type)
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,  # 测试集不需要shuffle
            drop_last=False,
            num_workers=getattr(args, 'num_workers', 4),
            pin_memory=True
        )
        
        return test_loader
    
    else:
        raise ValueError(f"不支持的模式: {args.mode}，请使用 'train' 或 'test'")

# def mtlfs_dataloader(args, prediction_type='both'):
#     """
#     MTLFA模型专用的数据加载器
    
#     Args:
#         args: 训练参数
#         prediction_type: 'both', 'short', 'long'
#     """
#     # 使用单一预测类型数据集
#     raw_dataset = MTLFS_PowerDataset(args, prediction_type)
    
#     if args.mode == 'train':
#         sampler = DistributedSampler(raw_dataset)
#         dataloader = DataLoader(raw_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
#     else:
#         dataloader = DataLoader(raw_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)

#     return dataloader