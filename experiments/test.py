import pandas as pd 
import numpy as np
import os
from pathlib import Path

driams_dir = Path(r"C:\Nguyen Tri\Paper\vi_khuan_khang_thuoc\DRIAMS\DRIAMS-A\binned_6000")

for item in driams_dir.iterdir():
    if item.name.startswith(".") or item.name == "desktop.ini":
        continue  # skip system files
    item_type = "Folder" if item.is_dir() else "File"
    print(f"[{item_type}] {item.name} -> Full path: {item}")
all_files = {}
for root, dirs, files in os.walk(driams_dir):
    for file_name in files:
        if file_name in all_files:
            all_files[file_name] += 1
            print("dupplicate rid" + file_name)
            continue
        all_files[file_name] = 1

