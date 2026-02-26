import os

def rename_images_in_folder(folder_path):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # 按照文件名排序
    files.sort()

    # 遍历文件并重命名
    for index, file in enumerate(files, start=1):
        new_name = f"driver_{index:02}.jpg"
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    folder_path = "data"  # 修改为你的文件夹路径
    rename_images_in_folder(folder_path)