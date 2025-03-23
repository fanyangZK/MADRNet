import os
# 同时添加数据清洗函数（可选）

def clean_hidden_files(data_dir):
    """删除数据集中的隐藏文件"""
    for root, dirs, files in os.walk(data_dir):
        # 删除隐藏文件夹
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        # 删除隐藏文件
        for f in files:
            if f.startswith('.'):
                os.remove(os.path.join(root, f))
                print(f"Removed hidden file: {os.path.join(root, f)}")