import re
import os
from tokenizers import Tokenizer
import argparse

def get_byte_mapping():
    """
    标准 ByteLevel 映射 (GPT-2 风格)，将字节 0-255 映射到 Unicode 字符
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

def is_chinese(char):
    """判断一个字符是否是汉字"""
    return '\u4e00' <= char <= '\u9fa5'

def export_readable_vocab(tokenizer_path: str, output_path: str):
    """
    将 BPE 分词器的词表导出为人类可读的文本文件，包含十六进制字节辅助调试
    """
    if not os.path.exists(tokenizer_path): return print(f"错误: 找不到分词器文件 {tokenizer_path}")
    tokenizer, byte_decoder = Tokenizer.from_file(tokenizer_path), get_byte_mapping()
    sorted_vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    print(f"正在导出词表到: {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Vocab Size: {len(sorted_vocab)}\n{'ID':<8} | {'Raw (BPE)':<25} | {'Hex Bytes':<25} | {'Decoded':<20}\n{'-'*90}\n")
        for t, tid in sorted_vocab:
            try: h = bytes([byte_decoder[c] for c in t]).hex(' ')
            except: h = "N/A"
            try:
                d = tokenizer.decode([tid])
                s = "[Fragment]" if not d.strip() and t.strip() else ("[Empty]" if not d and not t else d.replace('\n','\\n').replace('\r','\\r').replace('\t','\\t'))
            except: s = "[Error]"
            f.write(f"{tid:<8} | {t:<25} | {h:<25} | {s:<20}\n")
    print(f"导出完成！总计 {len(sorted_vocab)} 个 Token。")

def analyze_vocab_distribution(file_path):
    """
    分析已导出的可读词表文件中的 Token 分布
    """
    if not os.path.exists(file_path): return print(f"错误: 找不到文件 {file_path}")
    counts = {"chi": 0, "eng": 0, "oth": 0}
    print(f"正在分析词表分布: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 4 or any(line.startswith(p) for p in ['--', 'Vocab', 'ID']): continue
                decoded = parts[-1].strip()
                if any(is_chinese(c) for c in decoded): counts["chi"] += 1
                elif re.search(r'[a-zA-Z]', decoded): counts["eng"] += 1
                else: counts["oth"] += 1
        chinese_tokens, english_tokens, other_tokens = counts["chi"], counts["eng"], counts["oth"]
        total_tokens = sum(counts.values())
        if total_tokens == 0: return print("未找到有效的 Token 数据。")
    except Exception as e: print(f"分析过程中发生错误: {e}")
    print("\n" + "="*50)
    print(f"词表统计报告")
    print("="*50)
    print(f"Token 总数:   {total_tokens}")
    print("-" * 50)
    print(f"中文 Token:   {chinese_tokens:<8} 占比: {chinese_tokens/total_tokens:.2%}")
    print(f"英文 Token:   {english_tokens:<8} 占比: {english_tokens/total_tokens:.2%}")
    print(f"其他 Token:   {other_tokens:<8} 占比: {other_tokens/total_tokens:.2%}")
    print("="*50 + "\n")

if __name__ == '__main__':
    # 获取 src 目录路径 (当前文件在 src/data/tools/vocab_tool.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.dirname(os.path.dirname(current_dir))
    
    parser = argparse.ArgumentParser(description="词表分析与导出工具")
    parser.add_argument("--tokenizer", type=str, help="tokenizer.json 的路径")
    parser.add_argument("--output", type=str, help="导出的 vocab.txt 路径")
    parser.add_argument("--analyze", type=str, help="要分析的 vocab.txt 路径")
    
    args = parser.parse_args()
    
    if args.tokenizer:
        output_path = args.output or (os.path.join(os.path.dirname(args.tokenizer), "vocab.txt") if os.path.dirname(args.tokenizer) else "vocab.txt")
        export_readable_vocab(args.tokenizer, output_path)
        analyze_vocab_distribution(output_path)
    elif args.analyze:
        analyze_vocab_distribution(args.analyze)
    else:
        # 默认路径逻辑 (基于 src_path)
        DEFAULT_TOKENIZER = os.path.join(src_path, 'data', 'dataset', 'tokenizer', 'tokenizer.json')
        DEFAULT_OUTPUT = os.path.join(src_path, 'data', 'dataset', 'tokenizer', 'vocab.txt')
        
        if os.path.exists(DEFAULT_TOKENIZER):
            export_readable_vocab(DEFAULT_TOKENIZER, DEFAULT_OUTPUT)
            analyze_vocab_distribution(DEFAULT_OUTPUT)
        else:
            parser.print_help()
            print("\n[提示] 未提供参数且未找到默认路径的 tokenizer.json。")
