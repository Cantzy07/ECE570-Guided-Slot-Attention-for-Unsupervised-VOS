import os
import requests
import shutil
from tqdm import tqdm

# Define dataset URLs and target folder
DATASETS_TRAIN = {
    "duts_train": "http://saliencydetection.net/duts/download/DUTS-TR.zip",
    "davis_16": "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip",
    "fbms_train": "https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Trainingset.zip"
}
DATASETS_TEST = {
    "duts_test": "http://saliencydetection.net/duts/download/DUTS-TE.zip",
    "fbms_test": "https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Testset.zip"
}

def create_directory(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)  # Deletes the entire directory
        os.makedirs(path)  # Creates a new empty directory

def download_file(url, save_path):
    create_directory(save_path, overwrite=True)
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "wb") as file, tqdm(
        desc=save_path, total=file_size, unit="B", unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))

if __name__ == "__main__":
    TRAIN_DIR = "datasets/train"
    TEST_DIR = "datasets/test"
    for name, url in DATASETS_TRAIN.items():
        save_path = os.path.join(TRAIN_DIR, name + os.path.splitext(url)[1])
        download_file(url, save_path)
    for name, url in DATASETS_TEST.items():
        save_path = os.path.join(TEST_DIR, name + os.path.splitext(url)[1])
        download_file(url, save_path)
    