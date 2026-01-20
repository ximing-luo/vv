import os
import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
# 将项目根目录添加到 sys.path 以支持本地模块导入
root = str(Path(__file__).resolve().parents[2])
if root not in sys.path:
    sys.path.insert(0, root)
from configs.model import VVConfig, VisualVVConfig
from src.model import VV, VisualVV

def load_model(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载模型和分词器
    """
    if not os.path.exists(model_dir): raise FileNotFoundError(f"模型目录 {model_dir} 不存在")
    print(f"正在从 {model_dir} 加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = VisualVVConfig(
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        model = VisualVV(config)
        weights_path = os.path.join(model_dir, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location=device)
        print(f"成功加载权重: {weights_path}")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("模型加载完成。")
        return model, tokenizer, device
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        raise

def _smart_print(text, output_file=None, end="\n", flush=True):
    """
    内部辅助函数：同时打印到控制台和文件
    """
    print(text, end=end, flush=flush)
    if output_file:
        output_file.write(str(text) + end)
        output_file.flush()

def stream_inference(model, tokenizer, input_data, temperature=1.3, max_new_tokens=None, top_k=75, device='cpu', mode='chat', output_file=None, **kwargs):
    """ 流式推理生成文本 """
    max_new_tokens = max_new_tokens or model.config.max_seq_len
    pixel_values = None
    
    # 1. 构造输入与展示文本
    if mode == 'chat':
        full_prompt = tokenizer.apply_chat_template(input_data, tokenize=False, add_generation_prompt=True)
        user_prompt = input_data[-1]['content'] if isinstance(input_data, list) else input_data
        display_prompt = f"\nUser: {user_prompt}\nAssistant: "
    elif mode == 'pretrain':
        full_prompt = tokenizer.bos_token + input_data
        display_prompt = f"\n{input_data}"
    else: # vlm 模式
        image_path = kwargs.get('image_path')
        if not image_path: raise ValueError("Vision mode requires 'image_path' in kwargs")
        image = Image.open(image_path).convert('RGB')
        processor = CLIPImageProcessor.from_pretrained(model.config.vision_model_path)
        pixel_values = model.image2tensor(image, processor).unsqueeze(0).to(device)
        user_prompt = input_data[-1]['content'] if isinstance(input_data, list) else input_data
        full_prompt = f"{tokenizer.bos_token}User: {user_prompt.replace('<image>', model.config.image_special_token)}\nAssistant: "
        display_prompt = f"\nUser: {user_prompt}\nAssistant: "

    # 2. 编码并打印初始提示
    input_tensor = torch.tensor([tokenizer.encode(full_prompt, add_special_tokens=False)], dtype=torch.long).to(device)
    _smart_print(display_prompt, output_file, end="")
    
    # 3. 流式生成循环
    full_response, tokens_cached, printed_len = [], [], 0
    for next_token_tensor in model.generate_stream(input_tensor, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, pixel_values=pixel_values):
        token_id = next_token_tensor[0].item()
        if token_id == tokenizer.eos_token_id: break
        tokens_cached.append(token_id)
        full_text = tokenizer.decode(tokens_cached, skip_special_tokens=True)
        if full_text.endswith("\ufffd"): continue # 跳过 UTF-8 截断产生的乱码
        new_text = full_text[printed_len:]
        if new_text:
            _smart_print(new_text, output_file, end="", flush=True)
            full_response.append(new_text)
            printed_len = len(full_text)
            
    _smart_print("\n" + "-"*50, output_file)
    return ''.join(full_response)

def run_test_suite(model, tokenizer, device, mode, input_data, output_file, test_configs, max_new_tokens=100, image_path=None):
    """
    运行一组推理测试，测试不同的温度和 top_k 参数
    """
    for temp_val, tk_val, step, is_temp_fixed in test_configs:
        for i in range(4):
            temp = temp_val if is_temp_fixed else temp_val + i * step
            top_k = tk_val + i * step if is_temp_fixed else tk_val
            _smart_print(f"\n温度: {temp:.2f}, top_k: {int(top_k)}", output_file)
            for _ in range(2): # 每个配置运行两次
                stream_inference(model, tokenizer, input_data, output_file=output_file, max_new_tokens=max_new_tokens, mode=mode, device=device, temperature=temp, top_k=top_k, image_path=image_path)

def test():
    # 1. 获取模型根目录
    checkpoints_root = os.path.join(root, "models", "checkpoints")
    test_configs = [
        (1.3, 65, 5, True),  # 固定温度，变化 top_k
        (1.1, 75, 0.1, False) # 固定 top_k，变化温度
    ]
    messages = [{"role": "user", "content": "写一篇关于人工智能对未来发展的影响的文章。"}]
    prompt = (
        "    “咔咔！”\n"
        "    剧烈地疼痛从胸口处传来，叶晨勉力睁眼看去，只见眼前的世界一片血红，耳边，除了那带着几分兴奋的低沉兽吼，还有骨骼咀嚼的声音，令人毛骨悚然。\n"
        "    要死了吗？\n"
        "    叶晨心里有些苦涩，在末世里挣扎了十年之久，每天小心翼翼，连睡觉都是抱着兵器，稍有动静便会被惊动，今天却因为一个小小的疏忽，没有抹去猎杀三头犬时留下的气息，被这头血角兽给追踪上了。\n"
        "    就这样感受着身体被一点点嚼碎，也许是个不错的死法？\n"
    )
    image = ("请描述这张图片。<image>")    
    image_path = r".\src\data\database\gongjy\minimind-v_dataset\eval_images\彩虹瀑布-Rainbow-Falls .jpg"

    model_path = os.path.join(os.path.dirname(checkpoints_root), "vv")
    model, tokenizer, device = load_model(model_path)
    with open("inference_output.txt", "w", encoding="utf-8") as output_file:
        # # 2. 测试聊天模式
        print("\n=== 测试聊天模式 ===")
        run_test_suite(model, tokenizer, device, 'chat', messages, output_file, test_configs,
        max_new_tokens= 300
        )

        # 3. 测试续写模式
        print("\n=== 测试续写模式 ===")
        run_test_suite(model, tokenizer, device, 'pretrain', prompt, output_file, test_configs,
        max_new_tokens= 512
        )

        # 4. vlm测试
        print("\n=== 图片理解测试 ===")
        run_test_suite(model, tokenizer, device, 'vlm', image, output_file, test_configs,
        max_new_tokens= 200,
        image_path=image_path
        )

def main():
    print("="*30 + "\n  vv 模型推理工具\n" + "="*30)
    print("\n请选择推理模式:\n1. 聊天模式 (Chat)\n2. 续写模式 (Pretrain)\n3. 多模态模式 (VLM)")
    choice = input("\n请输入编号 (默认 1): ").strip()
    mode = 'chat' if choice == '1' else 'pretrain' if choice == '2' else 'vlm'
    
    model_path = os.path.join(root, "models", "vv")
    model, tokenizer, device = load_model(model_path)
    print(f"\n当前模式: {mode}\n输入 'q' 退出，'clear' 清空历史。")

    messages = []
    while True:
        try:
            image_path = None
            if mode == 'vlm':
                image_path = input("\n图片路径 > ").strip().strip('"\'')
                if not os.path.exists(image_path): continue
                prompt = input("User > ") or "描述这张图片<image>。"
            else:
                prompt = input(f"\n{'User' if mode == 'chat' else '续写'} > ")
            
            if prompt.lower() == 'q': break
            if prompt.lower() == 'clear': messages = []; continue
            if not prompt.strip(): continue
            
            input_data = messages + [{"role": "user", "content": prompt}] if mode == 'chat' else prompt
            res = stream_inference(model, tokenizer, input_data, device=device, mode=mode, image_path=image_path)
            if mode == 'chat': messages.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": res}])
        except KeyboardInterrupt: break
    print("\n推理结束。")

if __name__ == "__main__":
    test()
    # main()
