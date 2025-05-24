import os
import glob

# 指定目标文件夹路径
folder_path = "/home/syb/FineParser_cvproj/data/mask_heatmaps"

# 使用 glob 查找所有以 "result_" 开头的 .jpg 文件
files_to_delete = glob.glob(os.path.join(folder_path, "orig*.png"))

# 遍历并删除文件
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")