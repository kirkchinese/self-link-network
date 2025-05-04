"""
自链接模型预训练脚本
使用minimind的pretrain数据
"""

import os
import argparse
from train_selflink import SelfLinkTrainer

def main():
    parser = argparse.ArgumentParser(description='自链接模型预训练')
    
    # 训练参数
    parser.add_argument('--train_data_path', type=str, 
                       default='minimind\dataset\pretrain_hq.jsonl',
                       help='训练数据路径')
    parser.add_argument('--tokenizer_path', type=str, 
                       default='minimind/model/minimind_tokenizer',
                       help='tokenizer路径')
    parser.add_argument('--output_dir', type=str, default='output_pretrain',
                       help='模型输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='训练batch大小(默认32，显存不足时可减小)')
    parser.add_argument('--epochs', type=int, default=1,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大序列长度(默认512，显存不足时可减小)')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志打印间隔')
    parser.add_argument('--self_link_ratio', type=float, default=0.1,
                       help='自链接系数比例')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='每多少步保存一次检查点')
     
    args = parser.parse_args()
     
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
     
    trainer = SelfLinkTrainer(args)
    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch, args.save_steps)
     
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()