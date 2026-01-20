import os
import json
from utils.inference import stream_inference
from transformers import TrainerCallback

class InferenceCallback(TrainerCallback):
    """
    推理模拟回调：在评估后运行推理模拟，观察模型输出。
    支持多个测试案例。
    """
    def __init__(self, tokenizer, test_cases=None, **kwargs):
        self.tokenizer = tokenizer
        self.test_cases = test_cases

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        model.eval()
        # 获取设备
        device = next(model.parameters()).device
            
        for idx, case in enumerate(self.test_cases):
            prompt = case.get('prompt', "")
            mode = case.get('mode', 'pretrain')
            max_new_tokens = case.get('max_new_tokens', 512)
            temperature = case.get('temperature', 1.3)
            top_k = case.get('top_k', 75)
            image_path = case.get('image_path', None)
            
            print(f"\n{'='*20} 推理模拟 {idx+1}/{len(self.test_cases)} (Step: {state.global_step}, Mode: {mode}) {'='*20}")
            if mode == 'chat':
                # 对话模式：构造消息列表
                input_data = [{"role": "user", "content": prompt}]
            elif mode == 'vlm':
                # 视觉语言模型模式：直接使用字符串（stream_inference 会处理模板）
                input_data = prompt
            else:
                # 预训练模式：直接使用字符串
                input_data = prompt
            
            try:
                # 确定保存路径：保存到当前 checkpoint 目录下（如果存在），否则保存到 output_dir
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir, exist_ok=True)
                filename = f"{idx}_inference_{mode}.txt"
                save_path = os.path.join(checkpoint_dir, filename)
                
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(f"Step: {state.global_step}\n")
                    f.write(f"Mode: {mode}\n")
                    if metrics:
                        f.write(f"Metrics: {json.dumps(metrics, indent=2)}\n")
                    f.write(f"Prompt:\n{prompt}\n")
                    f.write(f"\n{'='*30}\n")

                    for _ in range(3):
                        for _ in stream_inference(
                            model, self.tokenizer, input_data, 
                            output_file=f, 
                            max_new_tokens=max_new_tokens, 
                            mode=mode, 
                            device=device, 
                            temperature=temperature, 
                            top_k=top_k,
                            image_path=image_path
                        ):
                            continue
                print(f"\n[System] 推理结果已保存至: {save_path}")
            except Exception as e:
                print(f"\n[Warning] 推理评估失败: {e}")
            print(f"\n{'='*50}\n")

        model.train()
