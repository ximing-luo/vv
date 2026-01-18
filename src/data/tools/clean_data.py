import os, glob, unicodedata
from concurrent.futures import ThreadPoolExecutor

def detect_encoding(fpath):
    for enc in ['utf-8', 'gb18030', 'big5']:
        try:
            with open(fpath, 'rb') as f:
                f.read(1024 * 100).decode(enc)
            return enc
        except: continue
    return None

def process_file(fpath):
    try:
        # 1. 编码转换
        enc = detect_encoding(fpath)
        if not enc: 
            print(f"[Deleted] 无法识别编码: {fpath}"); os.remove(fpath); return True
        if enc.lower() != 'utf-8':
            with open(fpath, 'r', encoding=enc) as f: content = f.read()
            with open(fpath, 'w', encoding='utf-8') as f: f.write(content)
            print(f"[Converted] {fpath} ({enc} -> utf-8)")
        # 2. 异常检测
        with open(fpath, 'r', encoding='utf-8') as f: content = f.read()
        reasons = []
        if '\0' in content: reasons.append("空字节/二进制")
        if '\ufffd' in content: reasons.append("损坏数据")
        if len(content) > 100:
            chi_ratio = len([c for c in content if '\u4e00' <= c <= '\u9fff']) / len(content)
            if chi_ratio < 0.1: reasons.append(f"汉字占比过低 ({chi_ratio:.1%})")
        if reasons:
            print(f"[Abnormal] {fpath} - {', '.join(reasons)} (删除)"); os.remove(fpath); return True
        return False
    except Exception as e:
        print(f"[Error] 处理 {fpath} 出错: {e}"); return False

def clean_data():
    # 获取动态路径
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(os.path.dirname(os.path.dirname(cur_dir)), "data", "database", "novel")
    print(f"扫描目录: {target_dir}")
    files = glob.glob(os.path.join(target_dir, "**/*.txt"), recursive=True)
    print(f"找到 {len(files)} 个文件，开始处理...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_file, files))
    print(f"处理完成。共删除 {sum(results)} 个异常文件。")

if __name__ == "__main__":
    clean_data()
