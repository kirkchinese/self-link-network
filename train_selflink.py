"""
自链接模型训练脚本

实现神经元自链接结构的实验性训练流程
"""

import os
import argparse
import time
import math
import warnings
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import SelfLinkDataset

# 忽略警告信息
warnings.filterwarnings('ignore')

class SelfLinkTrainer:
    """自链接模型训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        self.model = self._init_model()
        self.model.to(self.device)
        
        # 优化器和学习率调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器将在训练开始时初始化
        self.scheduler = None
        
        # 加载数据集
        self.train_dataset = SelfLinkDataset(
            args.train_data_path,
            self.tokenizer,
            max_length=args.max_length
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
    
    def _init_model(self):
        """初始化自链接模型"""
        from self_link_model import SelfLinkLM, SelfLinkConfig
        
        config = SelfLinkConfig(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=768,
            n_layer=12,
            n_head=12,
            self_link_ratio=self.args.self_link_ratio
        )
        
        return SelfLinkLM(config)
    
    def save_checkpoint(self, epoch, batch_idx, total_loss):
        """保存训练检查点
        Args:
            epoch: 当前epoch数
            batch_idx: 当前batch索引
            total_loss: 累计损失值
        """
        checkpoint_dir = os.path.join(self.args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_epoch{epoch}.pth'
        )
        
        torch.save({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': total_loss / (batch_idx + 1),
            'config': self.model.config
        }, checkpoint_path)
        
        print(f"检查点已保存至: {checkpoint_path}")

    def train(self, epoch=None, save_steps=1000):
        """训练循环
        Args:
            epoch: 当前epoch数
            save_steps: 每多少步保存一次检查点
        """
        self.model.train()
        
        # 初始化学习率调度器
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs * len(self.train_loader),
                eta_min=self.args.learning_rate * 0.1
            )
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 定期保存检查点
            if save_steps > 0 and (batch_idx + 1) % save_steps == 0:
                self.save_checkpoint(epoch or 1, batch_idx + 1, total_loss)
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'labels': batch['labels'].to(self.device)
            }
            
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = outputs['loss']
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  # 更新学习率
            
            total_loss += loss.item()
            
            if batch_idx % self.args.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                print(f'| epoch {self.args.epochs} | batch {batch_idx} | '
                      f'lr {self.optimizer.param_groups[0]["lr"]:.6f} | '
                      f'ms/batch {elapsed * 1000 / self.args.log_interval:.1f} | '
                      f'loss {avg_loss:.4f}')
                start_time = time.time()
    
    def save_model(self, output_dir):
        """保存模型"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存为pth格式
        model_path = os.path.join(output_dir, 'model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config
        }, model_path)
        
        # 单独保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"模型已保存至: {output_dir} (pth格式)")

def main():
    parser = argparse.ArgumentParser(description='自链接模型训练')
    
    # 训练参数
    parser.add_argument('--train_data_path', type=str, required=True,
                       help='训练数据路径')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                       help='tokenizer路径')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='模型输出目录')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='训练batch大小')
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志打印间隔')
    parser.add_argument('--self_link_ratio', type=float, default=0.1,
                       help='自链接系数比例')
    
    args = parser.parse_args()
    
    trainer = SelfLinkTrainer(args)
    for epoch in range(1, args.epochs + 1):
        trainer.train()
    
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()