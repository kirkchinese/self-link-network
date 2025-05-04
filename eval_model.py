import argparse
import torch
from transformers import AutoTokenizer
from self_link_model import SelfLinkLM, SelfLinkConfig

def load_model(model_path, tokenizer_path):
    """加载自链接模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 加载模型配置
    config = SelfLinkConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=768,
        n_layer=12,
        n_head=12,
        self_link_ratio=0.1
    )
    
    # 初始化并加载模型
    model = SelfLinkLM(config)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device).eval(), tokenizer

def interactive_test(model, tokenizer):
    """交互式测试模型"""
    print("自链接模型交互测试 (输入'quit'退出)")
    while True:
        prompt = input("\n👤 请输入: ")
        if prompt.lower() == 'quit':
            break
            
        # 编码输入
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        # 手动实现文本生成
        print("🤖 模型回复: ", end='', flush=True)
        generated = inputs['input_ids'].clone()
        max_length = inputs['input_ids'].shape[1] + 200
        
        with torch.no_grad():
            while generated.shape[1] < max_length:
                outputs = model(
                    input_ids=generated,
                    attention_mask=torch.ones_like(generated)
                )
                
                # 获取最后一个token的logits
                next_token_logits = outputs['logits'][:, -1, :]
                
                # 温度采样
                next_token_logits = next_token_logits / 0.9
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # top-p采样
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > 0.9
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[:, indices_to_remove] = 0
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)
                
                # 打印新生成的token
                new_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                print(new_text, end='', flush=True)
                
                # 遇到结束符则停止
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
        print()  # 换行

def main():
    parser = argparse.ArgumentParser(description='自链接模型评估脚本')
    parser.add_argument('--model_path', default='output_pretrain/model.pth',
                      help='模型文件路径')
    parser.add_argument('--tokenizer_path', default='output_pretrain',
                      help='tokenizer路径')
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.tokenizer_path)
    print(f"模型已加载，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    interactive_test(model, tokenizer)

if __name__ == "__main__":
    main()