import re
import os

# 词表文件路径
vocab_path = r'd:\Axon\ANN\llm\vv\minimind-master\model\vocab.txt'

def is_chinese(char):
    """判断一个字符是否是汉字"""
    return '\u4e00' <= char <= '\u9fa5'

def analyze_vocab_distribution(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    total_tokens = 0
    chinese_tokens = 0
    english_tokens = 0
    other_tokens = 0

    print(f"正在分析词表文件: {file_path} ...")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            # 跳过表头和分隔符行
            if not line or line.startswith('--') or line.startswith('Vocab') or line.startswith('ID'):
                continue
                
            parts = line.split('|')
            # 确保行格式正确 (至少有4列: ID, Raw, Hex, Decoded)
            if len(parts) < 4:
                continue
                
            # 最后一列是解码后的内容
            decoded = parts[-1].strip()
            
            # 统计逻辑：
            # 1. 只要包含汉字，就算作中文 Token
            # 2. 不含汉字但包含英文字母，算作英文 Token
            # 3. 其他算作符号/数字/特殊 Token
            if any(is_chinese(c) for c in decoded):
                chinese_tokens += 1
            elif re.search(r'[a-zA-Z]', decoded):
                english_tokens += 1
            else:
                other_tokens += 1
                
            total_tokens += 1

        if total_tokens == 0:
            print("未找到有效的 Token 数据。")
            return

        print("\n" + "="*40)
        print(f"词表统计报告")
        print("="*40)
        print(f"Token 总数:   {total_tokens}")
        print("-" * 40)
        print(f"中文 Token:   {chinese_tokens:<6} 占比: {chinese_tokens/total_tokens:.2%}")
        print(f"英文 Token:   {english_tokens:<6} 占比: {english_tokens/total_tokens:.2%}")
        print(f"其他 Token:   {other_tokens:<6} 占比: {other_tokens/total_tokens:.2%}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    analyze_vocab_distribution(vocab_path)
