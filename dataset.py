import json
import torch
from torch.utils.data import Dataset

class SelfLinkDataset(Dataset):
    """自链接模型训练数据集"""
    
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        参数:
            file_path: JSONL格式的数据文件路径
            tokenizer: 文本tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载JSONL文件
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item['text'])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize文本
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': inputs['input_ids'].squeeze(0)  # 语言模型使用输入作为标签
        }