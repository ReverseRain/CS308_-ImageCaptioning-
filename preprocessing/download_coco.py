import os
import argparse
import requests
import zipfile
import tarfile
import shutil
from tqdm import tqdm

def download_file(url, destination):
    """
    下载文件并显示进度条
    
    Args:
        url: 下载链接
        destination: 保存路径
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def extract_zip(zip_path, extract_path):
    """
    解压ZIP文件
    
    Args:
        zip_path: ZIP文件路径
        extract_path: 解压目标路径
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        print(f"解压 {zip_path} 到 {extract_path}...")
        for i, file in enumerate(zip_ref.infolist()):
            zip_ref.extract(file, extract_path)
            print(f"进度: {i+1}/{total_files}", end='\r')
    print()

def extract_tar(tar_path, extract_path):
    """
    解压TAR文件
    
    Args:
        tar_path: TAR文件路径
        extract_path: 解压目标路径
    """
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        total_files = len(tar_ref.getmembers())
        print(f"解压 {tar_path} 到 {extract_path}...")
        for i, file in enumerate(tar_ref.getmembers()):
            tar_ref.extract(file, extract_path)
            print(f"进度: {i+1}/{total_files}", end='\r')
    print()

def download_coco(data_dir):
    """
    下载COCO数据集
    
    Args:
        data_dir: 数据目录
    """
    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    coco_dir = os.path.join(data_dir, 'coco')
    os.makedirs(coco_dir, exist_ok=True)
    
    # 下载链接
    train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # 下载文件
    train_images_path = os.path.join(coco_dir, "train2017.zip")
    val_images_path = os.path.join(coco_dir, "val2017.zip")
    annotations_path = os.path.join(coco_dir, "annotations_trainval2017.zip")
    
    # 下载训练图像
    if not os.path.exists(train_images_path):
        print("下载训练图像...")
        download_file(train_images_url, train_images_path)
    else:
        print("训练图像已存在，跳过下载")
    
    # 下载验证图像
    if not os.path.exists(val_images_path):
        print("下载验证图像...")
        download_file(val_images_url, val_images_path)
    else:
        print("验证图像已存在，跳过下载")
    
    # 下载标注
    if not os.path.exists(annotations_path):
        print("下载标注...")
        download_file(annotations_url, annotations_path)
    else:
        print("标注已存在，跳过下载")
    
    # 解压文件
    extract_zip(train_images_path, coco_dir)
    extract_zip(val_images_path, coco_dir)
    extract_zip(annotations_path, coco_dir)
    
    print("COCO数据集下载和解压完成！")

def parse_args():
    parser = argparse.ArgumentParser(description="下载COCO数据集")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    download_coco(args.data_dir) 