"""
自链接模型全参数微调脚本
使用minimind的full_sft数据
"""

import os
import argparse
from train_selflink import SelfLinkTrainer

def main():
    parser = argparse.ArgumentParser(description='自链接模型全参数微调')
    
    # 训练参数
    parser.add_argument('--train_data_path', type=str, 
                       default='minimind/processed_data/processed_心理学训练试验.jsonl',
                       help='训练数据路径')
    parser.add_argument('--tokenizer_path', type=str, 
                       default='minimind/model/minimind_tokenizer',
                       help='tokenizer路径')
    parser.add_argument('--output_dir', type=str, default='output_full_sft',
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