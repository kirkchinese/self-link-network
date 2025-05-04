"""
自链接实验训练脚本
"""
import torch
from torch.utils.data import DataLoader
from minimind.model.dataset import PretrainDataset
from .self_link_experiment import SelfLinkLM
from .experiment_config import get_config
from tqdm import tqdm

def train_model(config=None, data_path="minimind/processed_data/processed_心理学训练试验.jsonl"):
    """训练自链接模型"""
    # 加载配置
    config = get_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = SelfLinkLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 加载数据
    dataset = PretrainDataset(data_path, tokenizer=None, max_length=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 训练循环
    for epoch in range(3):  # 3个epoch
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, targets, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits.view(-1, config.vocab_size), 
                           targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "learn-test/self_link_model.pt")
    return model

def compare_with_baseline():
    """与基线模型比较"""
    from minimind.model.model import MiniMindLM
    
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载两个模型
    self_link_model = SelfLinkLM(config).to(device)
    baseline_model = MiniMindLM(config).to(device)
    
    # 测试数据
    test_input = torch.randint(0, config.vocab_size, (1, 32)).to(device)
    
    # 比较推理速度
    import time
    start = time.time()
    _ = self_link_model(test_input)
    self_link_time = time.time() - start
    
    start = time.time()
    _ = baseline_model(test_input)
    baseline_time = time.time() - start
    
    print(f"自链接模型推理时间: {self_link_time:.4f}s")
    print(f"基线模型推理时间: {baseline_time:.4f}s")

if __name__ == "__main__":
    trained_model = train_model()
    if get_config().compare_baseline:
        compare_with_baseline()