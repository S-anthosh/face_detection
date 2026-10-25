from pathlib import Path
ROOT = Path(r"D:\VSB-TUO\machine learning\kaggle_cache\datasets\hereisburak\pins-face-recognition\versions\1\105_classes_pins_dataset")

print(ROOT.exists())                              # True varanum
folders = [d for d in ROOT.iterdir() if d.is_dir()]
print(len(folders))                               # 105 varanum
print(folders[0].name)                            # pins_... nu varum
print(len(list(folders[0].glob("*.jpg"))))        # images count