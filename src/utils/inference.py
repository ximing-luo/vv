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
        model.load_state_dict(state_dict, strict=False)
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
        for i in range(1):
            temp = temp_val if is_temp_fixed else temp_val + i * step
            top_k = tk_val + i * step if is_temp_fixed else tk_val
            _smart_print(f"\n温度: {temp:.2f}, top_k: {int(top_k)}", output_file)
            for _ in range(2): # 每个配置运行两次
                stream_inference(model, tokenizer, input_data, output_file=output_file, max_new_tokens=max_new_tokens, mode=mode, device=device, temperature=temp, top_k=top_k, image_path=image_path)

def run_vlm_batch_test(model, tokenizer, device, vlm_image_dir, vlm_prompts, output_file, test_configs):
    """
    批量测试 VLM 模型的图片理解能力，并测试不同参数配置
    """
    if not os.path.exists(vlm_image_dir):
        print(f"警告: VLM 测试目录 {vlm_image_dir} 不存在")
        return

    print("\n=== 开始批量 VLM 性能测试 ===")
    image_files = [f for f in os.listdir(vlm_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in image_files:
        img_path = os.path.join(vlm_image_dir, img_name)
        _smart_print(f"\n\n[VLM 测试] 图片: {img_name}", output_file)
        for v_prompt in vlm_prompts:
            _smart_print(f"\n>>> 提示词: {v_prompt}", output_file)
            # 使用传入的动态测试配置，而不再是固定值
            run_test_suite(model, tokenizer, device, 'vlm', v_prompt, output_file, test_configs, 
                         max_new_tokens=150, image_path=img_path)

def test():
    # 1. 获取模型根目录
    checkpoints_root = os.path.join(root, "models", "checkpoints")
    # test_configs = [
    #     (0.7, 30, 0.1, False),   # 低温稳定性测试：固定 top_k=30，温度从 0.7 开始递增
    #     (1.0, 50, 10, True),     # 均衡性测试：固定温度=1.0，top_k 从 50 开始递增
    #     (1.3, 80, -10, True)     # 高温多样性测试：固定温度=1.2，top_k 从 80 开始递减
    # ]
    test_configs = [(1.0, 55, 0.1, False)]
    # 对话模式提示词：涵盖逻辑、创意和常识
    chat_prompts = [
        [{"role": "user", "content": "我有3个苹果，吃掉1个后又买了2箱，每箱10个，现在我一共有多少个？"}],
        [{"role": "user", "content": "请写一段关于'赛博朋克风格的成都'的描写，要求富有画面感。"}],
        [{"role": "user", "content": "如果人工智能有了自我意识，它第一句话会说什么？"}]
    ]
    # 续写模式提示词：涵盖武侠、科幻和日常
    pretrain_prompts = [
        "　　方源一身残破的碧绿大袍，披头散发，浑身浴血，环顾四周。\n",
        "随着超空间引擎的轰鸣，巨大的星舰缓缓穿过虫洞，舷窗外，原本漆黑的宇宙被扭曲成了：",
        "清晨的阳光透过窗帘缝隙洒在桌上，咖啡的热气袅袅升起，他打开日记本，写下了第一行字："
    ]
    # VLM 测试：遍历 eval_images 目录下的所有图片
    vlm_image_dir = os.path.join(root, "src", "data", "database", "gongjy", "minimind-v_dataset", "eval_images")
    vlm_prompts = [
        "描述这张图片的内容。<image>",
        "图中有什么主要物体？<image>",
        "这张图给人的感觉是怎样的？<image>"
    ]
    
    model_path = os.path.join(os.path.dirname(checkpoints_root), "vv-1.3")
    model, tokenizer, device = load_model(model_path)
    with open("inference_output.txt", "w", encoding="utf-8") as output_file:
        # 1. 测试聊天模式
        print("\n=== 测试聊天模式 ===")
        for msg in chat_prompts:
            _smart_print(f"\n>>> 测试提示词: {msg[0]['content']}", output_file)
            run_test_suite(model, tokenizer, device, 'chat', msg, output_file, test_configs,
            max_new_tokens=200)

        # 2. 测试续写模式
        print("\n=== 测试续写模式 ===")
        test_configs = [(1.1, 45, 0.1, False)]
        for p in pretrain_prompts:
            _smart_print(f"\n>>> 测试提示词: {p}", output_file)
            run_test_suite(model, tokenizer, device, 'pretrain', p, output_file, test_configs,
            max_new_tokens=300)

        # 3. 批量 VLM 性能测试
        # 针对 VLM 调整测试配置：步长设为非零，以便观察参数变化
        # vlm_test_configs = [
        #     (0.8, 40, 0.1, False),  # 变化温度：0.8 -> 0.9 -> 1.0
        #     (0.9, 30, 20, True)     # 变化 Top-k：30 -> 50 -> 70
        # ]
        vlm_test_configs = [(0.9, 40, 0.1, False)]
        run_vlm_batch_test(model, tokenizer, device, vlm_image_dir, vlm_prompts, output_file, vlm_test_configs)

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
