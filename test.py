import os
os.environ["KAGGLEHUB_CACHE"] = r"D:\VSB-TUO\machine learning\kaggle_cache"

import kagglehub
path = kagglehub.dataset_download("hereisburak/pins-face-recognition")
print("Path:", path)
print(os.listdir(path))