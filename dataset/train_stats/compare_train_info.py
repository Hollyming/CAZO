import torch
import numpy as np

def compare_tensors(t1, t2, name, rtol=1e-5, atol=1e-8):
    """比较两个张量是否相近"""
    if isinstance(t1, (dict, list)) and isinstance(t2, (dict, list)):
        return compare_structures(t1, t2, name)
    
    if not torch.is_tensor(t1):
        t1 = torch.tensor(t1)
    if not torch.is_tensor(t2):
        t2 = torch.tensor(t2)
    
    if t1.shape != t2.shape:
        print(f"{name}: 形状不同 - {t1.shape} vs {t2.shape}")
        return False
    
    is_close = torch.allclose(t1.float(), t2.float(), rtol=rtol, atol=atol)
    if not is_close:
        print(f"{name}: 数值不同")
        print(f"最大差异: {torch.max(torch.abs(t1 - t2))}")
        return False
    return True

def compare_structures(d1, d2, prefix=""):
    """递归比较两个数据结构（字典或列表）"""
    if type(d1) != type(d2):
        print(f"{prefix}: 类型不同 - {type(d1)} vs {type(d2)}")
        return False
    
    if isinstance(d1, dict):
        if set(d1.keys()) != set(d2.keys()):
            print(f"{prefix}: 键不同")
            print(f"文件1独有的键: {set(d1.keys()) - set(d2.keys())}")
            print(f"文件2独有的键: {set(d2.keys()) - set(d1.keys())}")
            return False
        
        return all(compare_tensors(d1[k], d2[k], f"{prefix}.{k}" if prefix else k)
                  for k in d1.keys())
    
    elif isinstance(d1, list):
        if len(d1) != len(d2):
            print(f"{prefix}: 长度不同 - {len(d1)} vs {len(d2)}")
            return False
        
        return all(compare_tensors(v1, v2, f"{prefix}[{i}]")
                  for i, (v1, v2) in enumerate(zip(d1, d2)))
    
    return True

def main():
    # 加载两个.pt文件
    file1 = "train_info_adapter.pt"
    file2 = "train_info_prompts3.pt"
    
    print(f"正在比较文件:")
    print(f"1. {file1}")
    print(f"2. {file2}")
    print("-" * 50)
    
    data1 = torch.load(file1)
    data2 = torch.load(file2)
    
    # 比较数据
    is_identical = compare_structures(data1, data2)
    
    if is_identical:
        print("\n结论: 两个文件中的数据完全相同")
    else:
        print("\n结论: 两个文件中的数据存在差异")

if __name__ == "__main__":
    main() 