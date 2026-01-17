import os
import glob
import unicodedata
from concurrent.futures import ThreadPoolExecutor

def detect_encoding(fpath):
    """
    尝试检测文件编码。注意：移除 latin-1，因为它会错误地匹配几乎所有内容。
    """
    encodings = ['utf-8', 'gb18030', 'big5']
    for enc in encodings:
        try:
            with open(fpath, 'rb') as f:
                chunk = f.read(1024 * 100) # 读取更多以确保准确性
                chunk.decode(enc)
            return enc
        except:
            continue
    return None

def convert_to_utf8(fpath):
    try:
        enc = detect_encoding(fpath)
        if enc is None:
            print(f"[Deleted] 无法识别编码（疑似二进制或损坏）: {fpath}")
            os.remove(fpath)
            return

        if enc.lower() == 'utf-8':
            # 即使是 utf-8，也尝试 strict 读取以确认没有隐藏乱码
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    f.read()
                return # 已经是健康的 utf-8
            except UnicodeDecodeError:
                # 虽然声明是 utf-8 但实际包含非法字节，按非法处理
                pass
        
        # 严格读取，不使用 errors='replace'，确保转换质量
        try:
            with open(fpath, 'r', encoding=enc) as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"[Deleted] 声明为 {enc} 但实际包含非法序列: {fpath}")
            os.remove(fpath)
            return
        
        # 写入为 UTF-8
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"[Converted] {fpath} ({enc} -> utf-8)")
            
    except Exception as e:
        print(f"[Error] 处理文件 {fpath} 时出错: {e}")

def check_and_remove_abnormal(fpath):
    """
    检测文件是否异常（乱码、二进制等），如果是则删除
    """
    try:
        # 尝试以 UTF-8 读取
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"[Abnormal] {fpath} - 无法以 UTF-8 解码 (删除)")
            os.remove(fpath)
            return True
            
        # 检测 1: 包含空字节 (通常意味着二进制文件)
        if '\0' in content:
            print(f"[Abnormal] {fpath} - 包含空字节/二进制数据 (删除)")
            os.remove(fpath)
            return True
            
        # 检测 2: 包含任何替换字符 (U+FFFD)
        # 既然我们在转换时使用了严格模式，理论上不应该产生 FFFD。
        # 如果还有，说明原始文件可能就被损坏过，直接删除。
        if '\ufffd' in content:
            print(f"[Abnormal] {fpath} - 包含替换字符/损坏数据 (删除)")
            os.remove(fpath)
            return True
            
        # 检测 3: 中文字符比例检测 (预防 Mojibake)
        # 如果一个文件几乎没有中文字符，或者中文字符占比极低，对于小说语料来说是不正常的
        if len(content) > 100:
            chinese_chars = [c for c in content if '\u4e00' <= c <= '\u9fff']
            chinese_ratio = len(chinese_chars) / len(content)
            if chinese_ratio < 0.1: # 小说中汉字占比通常远高于 10%
                print(f"[Abnormal] {fpath} - 汉字占比过低 ({chinese_ratio:.1%}) (删除)")
                os.remove(fpath)
                return True
        
        return False
        
    except Exception as e:
        print(f"[Error] 检查文件 {fpath} 时出错: {e}")
        return False

def clean_data():
    target_dir = r"D:\Axon\ANN\llm\vv\src\data\database\novel"
    print(f"开始扫描目录: {target_dir}")
    
    # 递归查找所有 .txt 文件
    files = glob.glob(os.path.join(target_dir, "**/*.txt"), recursive=True)
    print(f"找到 {len(files)} 个文本文件...")
    
    # 1. 执行转换
    print("正在执行编码转换 (To UTF-8)...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # map 会立即执行
        list(executor.map(convert_to_utf8, files))
    
    # 2. 执行检查
    print("正在检查文件完整性...")
    abnormal_count = 0
    
    # 使用多线程加速检查
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check_and_remove_abnormal, files))
        
    abnormal_count = sum(results)
        
    print(f"检查完成。共发现并删除了 {abnormal_count} 个异常文件。")

if __name__ == "__main__":
    clean_data()
