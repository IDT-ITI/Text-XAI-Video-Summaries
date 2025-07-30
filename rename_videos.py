import os
import json

# Paths to your data folders
summe_folder = 'data/SumMe'
tvsum_folder = 'data/TVSum'
json_path = 'data/rename.json'

# Load renaming mapping
with open(json_path, "r") as f:
    rename_map = json.load(f)

# Map dataset names to their folder paths
dataset_paths = {
    "SumMe": summe_folder,
    "TVSum": tvsum_folder
}

def rename_videos(dataset_name, folder_path, mapping):
    print(f"\nRenaming videos in: {dataset_name}")
    for original_stem, new_stem in mapping.items():
        matched = False
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if not os.path.isfile(file_path):
                continue
            file_name, ext = os.path.splitext(file)
            if file_name == original_stem:
                new_file_path = os.path.join(folder_path, f"{new_stem}{ext}")
                os.rename(file_path, new_file_path)
                matched = True
                break
        if not matched:
            print(f"File not found for: {original_stem}")

# Rename videos for both datasets
for dataset, path in dataset_paths.items():
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        continue
    rename_videos(dataset, path, rename_map[dataset])
