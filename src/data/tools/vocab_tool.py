import re
import os
import json
from tokenizers import Tokenizer
import argparse
from typing import List, Union

def format_chat(messages):
    if not messages: return ""
    f = f"<|im_start|>系统\n{messages[0]['content']}<|im_end|>\n" if messages[0].get('role') == 'system' else "<|im_start|>系统\n你是一个有用的助手，由 Axon 开发。<|im_end|>\n"
    ms = messages[1:] if messages[0].get('role') == 'system' else messages
    for m in ms:
        r, c = m.get('role'), m.get('content', '')
        if r in ['user', 'assistant']: f += f"<|im_start|>{'用户' if r=='user' else '助手'}\n{c}<|im_end|>\n"
    return f

def prepare_cache_files(files: List[str], cache_dir: str) -> List[str]:
    os.makedirs(cache_dir, exist_ok=True); cfs = []
    for i, fp in enumerate(files):
        cf = os.path.join(cache_dir, f"cache_{i}.txt")
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f_in, open(cf, 'w', encoding='utf-8') as f_out:
            if not fp.endswith('.jsonl'): f_out.write(f_in.read())
            else:
                for ln in f_in:
                    if not (ln := ln.strip()): continue
                    try:
                        d = json.loads(ln)
                        if 'conversations' in d: f_out.write(format_chat(d['conversations']))
                        elif 'instruction' in d:
                            instr, out = d.get('instruction', ''), d.get('output', '')
                            msgs, ts = [], re.split(r'(Human:|Assistant:)', instr)
                            for j in range(1, len(ts), 2): msgs.append({"role": "user" if ts[j]=="Human:" else "assistant", "content": ts[j+1].strip()})
                            if msgs: msgs[-1]['content'] += out
                            else: msgs = [{"role": "user", "content": instr}, {"role": "assistant", "content": out}]
                            f_out.write(format_chat(msgs))
                        else: f_out.write(d.get('text', ln) + "\n")
                    except: f_out.write(ln + "\n")
        cfs.append(cf)
    return cfs

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
    分析已导出的可读词表文件中的 Token 分布，并将报告写入文件开头
    """
    if not os.path.exists(file_path): return print(f"错误: 找不到文件 {file_path}")
    counts = {"chi": 0, "eng": 0, "oth": 0}
    print(f"正在分析词表分布: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('|')
                if len(parts) < 4 or any(line.startswith(p) for p in ['--', 'Vocab', 'ID']): continue
                decoded = parts[-1].strip()
                if any(is_chinese(c) for c in decoded): counts["chi"] += 1
                elif re.search(r'[a-zA-Z]', decoded): counts["eng"] += 1
                else: counts["oth"] += 1
        chinese_tokens, english_tokens, other_tokens = counts["chi"], counts["eng"], counts["oth"]
        total_tokens = sum(counts.values())
        if total_tokens == 0: return print("未找到有效的 Token 数据。")
    except Exception as e: return print(f"分析过程中发生错误: {e}")

    report = (
        "\n" + "="*50 + "\n" +
        "词表统计报告\n" +
        "="*50 + "\n" +
        f"Token 总数:   {total_tokens}\n" +
        "-" * 50 + "\n" +
        f"中文 Token:   {chinese_tokens:<8} 占比: {chinese_tokens/total_tokens:.2%}\n" +
        f"英文 Token:   {english_tokens:<8} 占比: {english_tokens/total_tokens:.2%}\n" +
        f"其他 Token:   {other_tokens:<8} 占比: {other_tokens/total_tokens:.2%}\n" +
        "="*50 + "\n\n"
    )
    print(report)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report.strip() + "\n\n") # 写入报告
            f.writelines(lines) # 写入原内容
        print(f"报告已写入文件: {file_path}")
    except Exception as e: print(f"写入文件失败: {e}")

if __name__ == '__main__':
    # 获取 src 目录路径 (当前文件在 src/data/tools/vocab_tool.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.dirname(os.path.dirname(current_dir))
    
    parser = argparse.ArgumentParser(description="词表分析与导出工具")
    parser.add_argument("--tokenizer", type=str, help="tokenizer.json 的路径")
    parser.add_argument("--output", type=str, help="导出的 vocab.txt 路径")
    parser.add_argument("--analyze", type=str, help="要分析的 vocab.txt 路径")
    parser.add_argument("--test", action="store_true", help="测试 prepare_cache_files 逻辑")
    
    args = parser.parse_args()
    
    if args.test:
        test_dir = os.path.join(src_path, "data", "metadata")
        test_files = []
        # 寻找 SFT, Pretrain (Minimind, Novel) 的各一个文件进行测试
        for root, _, files in os.walk(test_dir):
            for f in files:
                if f.endswith(('.jsonl', '.txt')):
                    test_files.append(os.path.join(root, f))
                    break # 每个目录只取一个
        
        if not test_files: print("未找到测试文件"); exit()
        
        cache_dir = "tests/vocab_tool_test_cache"
        print(f"开始测试 prepare_cache_files, 选取文件: {test_files}")
        cfs = prepare_cache_files(test_files, cache_dir)
        
        for cf in cfs:
            print(f"\n--- 缓存文件内容预览: {cf} ---")
            with open(cf, 'r', encoding='utf-8') as f:
                for idx in range(1, 11):
                    line = f.readline()
                    if not line: break
                    # 显示行号，不跳过空行
                    content = line.strip()
                    print(f"L{idx}: {content[:100] + ('...' if len(content)>100 else '')}")
        
        import shutil
        shutil.rmtree(cache_dir)
        print("\n测试完成，缓存已清理。")
    elif args.tokenizer:
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
