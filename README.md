# SelfLink Language Model (自链接语言模型)

## 项目概述
本项目实现了一种结合自注意力机制和自链接机制的增强型语言模型，通过神经元间的自链接结构增强模型表达能力，同时保持Transformer架构的高效性。

## 模型架构
- **混合注意力层(HybridAttentionLayer)**
  - 结合标准自注意力和自链接机制
  - 12头注意力设计
  - 残差连接和层归一化

- **增强版自链接层(EnhancedSelfLinkLayer)**
  - 多头自链接设计(默认比例0.1)
  - 独立的前馈网络处理每个头

- **旋转位置编码(RoPE)**
  - 相对位置编码实现
  - 支持长序列建模

- **基础架构**
  - 12层Transformer结构
  - 768维隐藏层
  - 1.2亿参数

## 创新点
1. **自链接机制**
   - 可调节的自链接比例(self_link_ratio)
   - 增强神经元间的局部连接

2. **混合注意力设计**
   - 自注意力与自链接的协同工作
   - 提升模型表达能力和训练稳定性

3. **优化的训练流程**
   - 余弦退火学习率调度
   - 定期检查点保存
   - 支持从断点恢复训练

## 训练流程
### 预训练
```bash
python pretrain.py \
  --train_data_path minimind/dataset/pretrain_hq.jsonl \
  --tokenizer_path minimind/model/minimind_tokenizer \
  --output_dir output_pretrain \
  --batch_size 32 \
  --epochs 3 \
  --self_link_ratio 0.1
```

### 微调训练
```bash
python train_selflink.py \
  --train_data_path your_data.jsonl \
  --tokenizer_path output_pretrain \
  --output_dir output_finetune \
  --batch_size 8 \
  --epochs 5 \
  --self_link_ratio 0.05
```

## 评估方法
```bash
python eval_model.py \
  --model_path output_pretrain/model.pth \
  --tokenizer_path output_pretrain
```
- **交互式测试**：支持与模型对话测试
- **生成策略**：top-p采样(p=0.9) + 温度采样(t=0.9)
- **待实现**：困惑度等量化指标

## 优缺点分析
### 优点
- 自链接机制增强了模型表达能力
- 混合注意力设计提高了训练稳定性
- 参数效率较高(1.2亿参数)
- 支持长序列建模(最大1024 tokens)

### 缺点/限制
- 目前仅支持交互式评估
- 缺乏量化评估指标
- 自链接比例需要精细调优
- 显存消耗较大(全精度训练)
- 神经网络训练收敛困难
- 评估时偶发卡死问题(原因待查)

## 使用说明
1. **环境准备**
```bash
pip install -r requirements.txt
```

2. **数据准备**
- 预训练数据：minimind/dataset/pretrain_hq.jsonl
- 微调数据：自定义jsonl格式

3. **训练**
- 参见上方训练流程部分

4. **评估**
- 交互式测试：`python eval_model.py`
- 量化评估：待实现

## 数据来源
本项目使用[minimind项目](https://github.com/jingyaogong/minimind)提供的高质量预训练数据集，特别感谢minimind项目组的贡献。

## 未来改进方向
1. 实现量化评估指标(困惑度等)
2. 添加模型压缩支持(量化、蒸馏)
3. 优化显存使用(混合精度训练)
4. 扩展自链接机制的应用场景
5. 解决训练收敛问题
6. 修复评估卡死问题