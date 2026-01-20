import os
import sys
current_file_path = os.path.abspath(__file__) 
src_path = os.path.dirname(os.path.dirname(current_file_path))
vv_path = os.path.dirname(src_path)
project_root = os.path.dirname(vv_path)
for path in [src_path, vv_path, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from PIL import Image
from configs.model import VVConfig, VisualVVConfig
from model.model import VV
from model.model_vlm import VisualVV
from transformers import AutoTokenizer, CLIPImageProcessor

def load_model(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载模型和分词器
    """
    if not os.path.exists(model_dir): raise FileNotFoundError(f"模型目录 {model_dir} 不存在")
    print(f"正在从 {model_dir} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
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
        for i in range(8):
            temperature = temp_val if is_temp_fixed else temp_val + i * step
            top_k = tk_val + i * step if is_temp_fixed else tk_val
            
            info = f"\n温度: {temperature:.2f}, top_k: {int(top_k) if is_temp_fixed else top_k}"
            print(info)
            output_file.write(info + "\n")
            output_file.flush()
            
            # 每个配置运行两次生成
            for _ in range(2):
                stream_inference(
                    model, tokenizer, input_data, 
                    output_file=output_file, 
                    max_new_tokens=max_new_tokens, 
                    mode=mode, 
                    device=device, 
                    temperature=temperature, 
                    top_k=top_k,
                    image_path=image_path
                )

def test():
    # 1. 获取模型根目录
    checkpoints_root = os.path.join(vv_path, "models", "checkpoints")
    test_configs = [
        (1.3, 40, 5, True),  # 固定温度，变化 top_k
        (0.8, 75, 0.1, False) # 固定 top_k，变化温度
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
    image_path = r"D:\Axon\ANN\llm\vv\src\data\database\gongjy\minimind-v_dataset\eval_images\彩虹瀑布-Rainbow-Falls .jpg"

    model_path = os.path.join(os.path.dirname(checkpoints_root), "vv")
    model, tokenizer, device = load_model(model_path)
    with open("inference_output.txt", "w", encoding="utf-8") as output_file:
        # # 2. 测试聊天模式
        # print("\n=== 测试聊天模式 ===")
        # run_test_suite(model, tokenizer, device, 'chat', messages, output_file, test_configs,
        # max_new_tokens= 400
        # )

        # 3. 测试续写模式
        # print("\n=== 测试续写模式 ===")
        # run_test_suite(model, tokenizer, device, 'pretrain', prompt, output_file, test_configs,
        # max_new_tokens= 2048
        # )

        # 4. vlm测试
        print("\n=== 图片理解测试 ===")
        run_test_suite(model, tokenizer, device, 'vlm', image, output_file, test_configs,
        max_new_tokens= 200,
        image_path=image_path
        )

def main():
    # 1. 获取模型根目录
    models_path = os.path.join(vv_path, "models")
    checkpoints_root = os.path.join(models_path, "checkpoints")
    
    print("="*30)
    print("  vv 模型推理工具")
    print("="*30)
    
    # 2. 选择模式
    print("\n请选择推理模式:")
    print("1. 聊天模式 (Chat) - 自动加载微调模型")
    print("2. 续写模式 (Pretrain) - 自动加载预训练模型")
    print("3. 多模态模式 (VLM) - 自动加载 VLM 模型")
    
    choice = input("\n请输入编号 (默认 1): ").strip()
    mode = 'chat' if choice == '1' else 'pretrain' if choice == '2' else 'vlm'
    
    # 3. 自动查找模型
    model_path = os.path.join(os.path.dirname(checkpoints_root), "vv")
    if not model_path:
        print(f"\n[错误] 在 {checkpoints_root} 下找不到模型权重。")
        sys.exit(1)
    
    # 加载
    try:
        model, tokenizer, device = load_model(model_path)
    except Exception as e:
        print(f"\n[加载失败] {e}")
        sys.exit(1)
    
    print(f"\n当前激活模式: {'聊天模式' if mode == 'chat' else '续写模式' if mode == 'pretrain' else '多模态模式'}")
    print("输入 'q' 退出，输入 'clear' 清空对话历史。")

    # 循环对话
    messages = []
    while True:
        try:
            if mode == 'chat':
                prompt = input("\nUser > ")
            elif mode == 'pretrain':
                prompt = input("\n续写输入 > ")
            else:
                # 多模态模式：处理图片和文本
                image_path = input("\n请输入图片路径 (绝对路径): ").strip().strip('"').strip("'")
                if not os.path.exists(image_path):
                    print(f"\n[错误] 图片路径 {image_path} 不存在。")
                    continue
                prompt = input("\nUser > ")
                if not prompt: prompt = "描述这张图片<image>。"
                
            if prompt.lower() == 'q': break
            if prompt.lower() == 'clear':
                messages = []
                print("对话历史已清空。")
                continue
            if not prompt.strip():
                continue
            
            if mode == 'chat':
                # 聊天模式：累加历史消息
                messages.append({"role": "user", "content": prompt})
                full_response = stream_inference(model, tokenizer, messages, device=device, mode=mode)
                messages.append({"role": "assistant", "content": full_response})
            elif mode == 'pretrain':
                # 续写模式：确保每次都是独立的输入，不带任何历史和标记
                stream_inference(model, tokenizer, prompt, temperature=1.3, top_k=75, device=device, mode=mode)
            else:
                # 多模态模式：处理图片和文本
                stream_inference(model, tokenizer, prompt, temperature=1.3, top_k=75, device=device, mode=mode, image_path=image_path)
            
        except KeyboardInterrupt:
            break
    
    print("\n推理结束。")

if __name__ == "__main__":
    # test()
    main()
