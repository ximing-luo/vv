import os
import sys
import torch

# 设置路径，确保可以导入项目中的模块
# 我们将 AGI 所在的目录添加到 sys.path
current_file_path = os.path.abspath(__file__) # .../AGI/src/utils/inference.py
src_path = os.path.dirname(os.path.dirname(current_file_path)) # .../AGI/src
agi_path = os.path.dirname(src_path) # .../AGI
project_root = os.path.dirname(agi_path) # .../ (d:\Axon\ANN\llm)

for path in [src_path, agi_path, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)
from configs.model import VVConfig
from model.model import VV
from transformers import AutoTokenizer

def load_model(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    加载模型和分词器
    """
    print(f"正在从 {model_dir} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = VVConfig(
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 3. 初始化模型
    model = VV(config)
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location=device)
    print(f"成功加载权重: {weights_path}")
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("模型加载完成。")
    return model, tokenizer, device

config = VVConfig()
def stream_inference(model, tokenizer, input_data, temperature=1.3, max_new_tokens=config.max_seq_len, top_k=75, device='cpu', mode='chat', output_file=None):
    """
    流式推理生成文本
    :param input_data: 如果是 chat 模式，为 messages 列表；如果是 pretrain 模式，为 prompt 字符串
    :param mode: 'pretrain' (续写模式) 或 'chat' (对话模式)
    :param output_file: 可选的文件对象，用于同步记录输出
    """
    def smart_print(text, end="\n", flush=True):
        print(text, end=end, flush=flush)
        if output_file:
            output_file.write(str(text) + end)
            output_file.flush()

    # 根据模式构造输入文本
    if mode == 'chat':
        full_prompt = tokenizer.apply_chat_template(input_data, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = tokenizer.bos_token + input_data

    # 编码输入
    input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    if mode == 'chat':
        # 仅打印当前输入的 prompt
        user_prompt = input_data[-1]['content'] if isinstance(input_data, list) else input_data
        smart_print(f"\nUser: {user_prompt}")
        smart_print("Assistant: ", end="")
    else:
        # 预训练/续写模式
        smart_print(f"\n{input_data}", end="")
    
    full_response = []
    tokens_cached = []
    printed_len = 0
    # 调用模型的流式生成方法
    for next_token_tensor in model.generate_stream(
        input_tensor, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k
    ):
        token_id = next_token_tensor[0].item()
        
        if token_id == tokenizer.eos_token_id:
            break
        
        tokens_cached.append(token_id)
        # 解码整个序列
        full_text = tokenizer.decode(tokens_cached, skip_special_tokens=True)
        
        # 处理 UTF-8 截断产生的乱码字符 (\ufffd)
        if full_text and full_text.endswith("\ufffd"):
            continue
            
        # 计算新生成的文本内容
        new_text = full_text[printed_len:]
        if new_text:
            smart_print(new_text, end="", flush=True)
            full_response.append(new_text)
            printed_len = len(full_text)
            
    smart_print("\n" + "-"*50)
    return ''.join(full_response)

def inference(model, tokenizer, input_data, temperature=1.3, max_new_tokens=config.max_seq_len, top_k=75, device='cpu', mode='chat', output_file=None):
    """
    非流式推理生成文本：等待所有 token 生成完成后一次性解码并返回。
    :param input_data: 如果是 chat 模式，为 messages 列表；如果是 pretrain 模式，为 prompt 字符串
    :param mode: 'pretrain' (续写模式) 或 'chat' (对话模式)
    :param output_file: 可选的文件对象，用于同步记录输出
    """
    def smart_print(text, end="\n", flush=True):
        print(text, end=end, flush=flush)
        if output_file:
            output_file.write(str(text) + end)
            output_file.flush()

    # 1. 构造输入
    if mode == 'chat':
        full_prompt = tokenizer.apply_chat_template(input_data, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = tokenizer.bos_token + input_data

    input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 2. 打印提示信息
    if mode == 'chat':
        user_prompt = input_data[-1]['content'] if isinstance(input_data, list) else input_data
        smart_print(f"\nUser: {user_prompt}")
        smart_print("Assistant: ", end="")
    else:
        smart_print(f"\n{input_data}", end="")

    # 3. 循环生成 token (不实时解码)
    generated_tokens = []
    for next_token_tensor in model.generate_stream(
        input_tensor, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k
    ):
        token_id = next_token_tensor[0].item()
        if token_id == tokenizer.eos_token_id:
            break
        generated_tokens.append(token_id)
    
    # 4. 一次性全量解码
    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 5. 一次性渲染输出
    smart_print(full_response)
    smart_print("-" * 50)

def find_latest_model(checkpoints_root, mode):
    """
    根据模式直接使用对应的 final 模型。
    mode='chat' -> finetune/final
    mode='pretrain' -> pretrain/final
    """
    subfolder = "finetune" if mode == 'finetune' else "pretrain"
    final_path = os.path.join(checkpoints_root, subfolder, "final")
    
    # 检查是否存在
    if os.path.exists(final_path):
        has_bin = os.path.exists(os.path.join(final_path, "pytorch_model.bin"))
        if has_bin:
            return final_path
            
    return None

def run_test_suite(model, tokenizer, device, mode, input_data, output_file, test_configs, max_new_tokens=100):
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
                    top_k=top_k
                )

def test():
    # 1. 获取模型根目录
    checkpoints_root = os.path.join(agi_path, "checkpoints")
    test_configs = [
        (1.3, 5, 10, True),  # 固定温度，变化 top_k
        (0.5, 75, 0.2, False) # 固定 top_k，变化温度
    ]
    messages = [{"role": "user", "content": "写一篇关于人工智能对未来发展的影响的文章。"}]
    prompt = (
        "    “咔咔！”\n"
        "    剧烈地疼痛从胸口处传来，叶晨勉力睁眼看去，只见眼前的世界一片血红，耳边，除了那带着几分兴奋的低沉兽吼，还有骨骼咀嚼的声音，令人毛骨悚然。\n"
        "    要死了吗？\n"
        "    叶晨心里有些苦涩，在末世里挣扎了十年之久，每天小心翼翼，连睡觉都是抱着兵器，稍有动静便会被惊动，今天却因为一个小小的疏忽，没有抹去猎杀三头犬时留下的气息，被这头血角兽给追踪上了。\n"
        "    就这样感受着身体被一点点嚼碎，也许是个不错的死法？\n"
    )
    with open("inference_output.txt", "w", encoding="utf-8") as output_file:
        # 2. 测试聊天模式
        print("\n=== 测试聊天模式 ===")
        model_path = find_latest_model(checkpoints_root, 'finetune')
        model, tokenizer, device = load_model(model_path)
        run_test_suite(model, tokenizer, device, 'chat', messages, output_file, test_configs,
        max_new_tokens= 400
        )

        # 3. 测试续写模式
        # print("\n=== 测试续写模式 ===")
        # model_path = find_latest_model(checkpoints_root, 'finetune')
        # model, tokenizer, device = load_model(model_path)
        # run_test_suite(model, tokenizer, device, 'pretrain', prompt, output_file, test_configs,
        # max_new_tokens= 2048
        # )

def main():
    # 1. 获取模型根目录
    checkpoints_root = os.path.join(agi_path, "checkpoints")
    
    print("="*30)
    print("  AGI 模型推理工具")
    print("="*30)
    
    # 2. 选择模式
    print("\n请选择推理模式:")
    print("1. 聊天模式 (Chat) - 自动加载微调模型")
    print("2. 续写模式 (Pretrain) - 自动加载预训练模型")
    
    choice = input("\n请输入编号 (默认 1): ").strip()
    mode = 'pretrain' if choice == '2' else 'chat'
    
    # 3. 自动查找模型
    model_path = find_latest_model(checkpoints_root, mode)
    
    if not model_path:
        print(f"\n[错误] 在 {checkpoints_root} 下找不到 {mode} 模式的有效模型。")
        print("请检查目录结构是否包含 'final' 或 'checkpoint-N' 子目录，且其中有权重文件。")
        sys.exit(1)
    
    print(f"\n[自动发现] 匹配到模型: {model_path}")
    
    # 加载
    try:
        model, tokenizer, device = load_model(model_path)
    except Exception as e:
        print(f"\n[加载失败] {e}")
        sys.exit(1)
    
    print(f"\n当前激活模式: {'聊天模式' if mode == 'chat' else '续写模式'}")
    print("输入 'q' 退出，输入 'clear' 清空对话历史。")

    # 循环对话
    messages = []
    while True:
        try:
            if mode == 'chat':
                prompt = input("\nUser > ")
            else:
                prompt = input("\n续写输入 > ")
                
            if prompt.lower() == 'q':
                break
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
            else:
                # 续写模式：确保每次都是独立的输入，不带任何历史和标记
                stream_inference(model, tokenizer, prompt, temperature=1.3, top_k=75, device=device, mode=mode)
            
        except KeyboardInterrupt:
            break
    
    print("\n推理结束。")

if __name__ == "__main__":
    test()
    # main()
