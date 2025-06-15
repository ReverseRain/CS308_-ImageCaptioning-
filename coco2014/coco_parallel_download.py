import os
import threading
import requests
from tqdm import tqdm

COCO_URLS = [
    ("train2014.zip", "http://images.cocodataset.org/zips/train2014.zip"),
    ("val2014.zip", "http://images.cocodataset.org/zips/val2014.zip"),
    ("annotations_trainval2014.zip", "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"),
]

CHUNK_SIZE = 1024 * 1024  # 1MB
NUM_THREADS = 8  # 可根据带宽调整

def get_file_size(url):
    r = requests.head(url, allow_redirects=True)
    return int(r.headers.get('content-length', 0))


def download_chunk(url, start, end, file_path, pbar):
    headers = {'Range': f'bytes={start}-{end}'}
    r = requests.get(url, headers=headers, stream=True)
    with open(file_path, 'r+b') as f:
        f.seek(start)
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def parallel_download(url, file_path, num_threads=NUM_THREADS):
    file_size = get_file_size(url)
    if os.path.exists(file_path):
        first_size = os.path.getsize(file_path)
    else:
        first_size = 0
    if first_size < file_size:
        with open(file_path, 'ab') as f:
            f.truncate(file_size)
    pbar = tqdm(total=file_size, initial=first_size, unit='B', unit_scale=True, desc=file_path)
    threads = []
    part = file_size // num_threads
    for i in range(num_threads):
        start = i * part
        end = file_size - 1 if i == num_threads - 1 else (i + 1) * part - 1
        t = threading.Thread(target=download_chunk, args=(url, start, end, file_path, pbar))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    pbar.close()

def main():
    for fname, url in COCO_URLS:
        print(f"开始下载: {fname}")
        parallel_download(url, fname)
        print(f"下载完成: {fname}\n")
    print("全部下载完成！")

if __name__ == "__main__":
    main() 