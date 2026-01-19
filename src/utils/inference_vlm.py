import os
import sys
import torch
import argparse
from PIL import Image

# 设置路径
current_file_path = os.path.abspath(__file__)
src_path = os.path.dirname(os.path.dirname(current_file_path))
vv_path = os.path.dirname(src_path)
project_root = os.path.dirname(vv_path)
for path in [src_path, vv_path, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from configs.model import VisualVVConfig
from model.model_vlm import VisualVV
from transformers import AutoTokenizer

def load_vlm_model(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载 VLM 模型、分词器和视觉处理器
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录 {model_dir} 不存在")
        
    print(f"正在从 {model_dir} 加载 VLM 模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # 加载配置
    config = VisualVVConfig(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 初始化模型
    model = VisualVV(config)
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    
    try:
        state_dict = torch.load(weights_path, map_location=device)
        print(f"成功加载权重: {weights_path}")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("VLM 模型加载完成。")
        return model, tokenizer, device
    except Exception as e:
        print(f"加载 VLM 模型权重时出错: {e}")
        raise

def _smart_print(text, output_file=None, end="\n", flush=True):
    print(text, end=end, flush=flush)
    if output_file:
        output_file.write(str(text) + end)
        output_file.flush()

def stream_vlm_inference(model, tokenizer, prompt, image_path, temperature=1.0, top_k=50, max_new_tokens=512, device='cpu'):
    """
    流式 VLM 推理
    """
    # 1. 处理图像
    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 {image_path}")
        return ""
        
    image = Image.open(image_path).convert('RGB')
    pixel_values = model.image2tensor(image, model.processor).unsqueeze(0).to(device) # (1, 1, 3, 224, 224)
    # 注意：VisualVV._apply_vision_embeddings 期望 pixel_values 形状为 (bs, num, c, h, w)
    # 这里的 num 是图片数量，我们这里只用一张图片。
    
    # 2. 构造 Prompt
    # 在 VLM 中，图像占位符通常放在开头，后面紧跟对话
    # 根据 VisualVVConfig，image_special_token 是 196 个 '@'
    image_placeholder = model.config.image_special_token
    # 按照训练时的模板构造输入
    full_prompt = f"{tokenizer.bos_token}{image_placeholder}\nUser: {prompt}\nAssistant: "
    
    # 3. 编码
    input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print("\nAssistant: ", end="", flush=True)
    
    full_response = []
    tokens_cached = []
    printed_len = 0
    
    # 4. 生成
    with torch.no_grad():
        for next_token_tensor in model.generate_stream(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            pixel_values=pixel_values
        ):
            token_id = next_token_tensor[0].item()
            if token_id == tokenizer.eos_token_id:
                break
                
            tokens_cached.append(token_id)
            full_text = tokenizer.decode(tokens_cached, skip_special_tokens=True)
            
            if full_text and full_text.endswith("\ufffd"):
                continue
                
            new_text = full_text[printed_len:]
            if new_text:
                print(new_text, end="", flush=True)
                full_response.append(new_text)
                printed_len = len(full_text)
                
    print("\n" + "-"*50)
    return "".join(full_response)

def main():
    parser = argparse.ArgumentParser(description="VV VLM 推理工具")
    parser.add_argument("--image", type=str, help="图片路径")
    parser.add_argument("--prompt", type=str, default="描述这张图片。", help="提示词")
    parser.add_argument("--model_path", type=str, default=os.path.join(vv_path, "models", "vv"), help="模型路径")
    args = parser.parse_args()

    model_path = args.model_path
    
    print("="*30)
    print("  vv VLM 推理工具")
    print("="*30)
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型目录 {model_path}")
        sys.exit(1)
        
    print(f"使用模型: {model_path}")
    
    try:
        model, tokenizer, device = load_vlm_model(model_path)
    except Exception as e:
        print(f"加载失败: {e}")
        sys.exit(1)

    # 如果命令行传入了图片，直接进行单次推理
    if args.image:
        print(f"\n--- 单次推理 ---")
        print(f"图片: {args.image}")
        print(f"User: {args.prompt}")
        stream_vlm_inference(model, tokenizer, args.prompt, args.image, device=device)
        return

    print("\n--- 交互模式 ---")
    print("输入 'q' 退出。")
    
    while True:
        try:
            image_path = input("\n请输入图片路径 > ").strip()
            if image_path.lower() == 'q': break
            if not image_path: continue
            if not os.path.exists(image_path):
                print(f"文件不存在: {image_path}")
                continue
                
            prompt = input("User > ").strip()
            if prompt.lower() == 'q': break
            if not prompt: prompt = "描述这张图片。"
            
            stream_vlm_inference(model, tokenizer, prompt, image_path, device=device)
            
        except KeyboardInterrupt:
            break
            
    print("\n推理结束。")

if __name__ == "__main__":
    main()
