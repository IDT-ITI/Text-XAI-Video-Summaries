import os
import pandas as pd

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "data")

# Define output folder inside the current directory
output_dir = os.path.join(current_dir, "final_scores")
os.makedirs(output_dir, exist_ok=True)

# Exclusions
summe_excluded_top_1 = {1, 3, 5, 11, 13, 16, 20, 24, 25}  # only for top-1
tvsum_excluded_top_1 = {3, 15, 21}
tvsum_excluded_top_3 = {1, 3, 6, 12, 15, 21, 29, 37, 39, 41, 46}

# Column groups
top_1_cols = [
    "sbert_attention_top_1_sum",
    "simcse_attention_top_1_sum",
    "sbert_lime_top_1_sum",
    "simcse_lime_top_1_sum",
]

top_3_cols = [
    "sbert_attention_sum",
    "simcse_attention_sum",
    "sbert_lime_sum",
    "simcse_lime_sum",
    "sbert_alternative_attention_sum",
    "simcse_alternative_attention_sum",
    "sbert_alternative_lime_sum",
    "simcse_alternative_lime_sum",
]


def get_video_index(video_name: str) -> int:
    """Extract the integer index from a folder like 'video_10'."""

    try:
        return int(video_name.split("_")[1])
    except Exception:
        return -1


def collect_csvs(dataset_name):
    """Find all CSV files inside text_explanation folders of a dataset."""

    dataset_path = os.path.join(data_path, dataset_name)
    csv_files = []
    for video_folder in os.listdir(dataset_path):
        video_path = os.path.join(dataset_path, video_folder)
        if not os.path.isdir(video_path):
            continue
        explanation_path = os.path.join(video_path, "text_explanation")
        if not os.path.isdir(explanation_path):
            continue
        for file in os.listdir(explanation_path):
            if file.endswith(".csv"):
                csv_files.append((video_folder, os.path.join(explanation_path, file)))
    return csv_files


def compute_average_top_1():
    """Compute averages for top_1 separately for SumMe and TVSum with exclusions."""

    all_results = []
    for dataset in ["SumMe", "TVSum"]:
        values = []
        csvs = collect_csvs(dataset)
        for video_folder, csv_file in csvs:
            video_idx = get_video_index(video_folder)
            if dataset == "SumMe" and video_idx in summe_excluded_top_1:
                continue
            if dataset == "TVSum" and video_idx in tvsum_excluded_top_1:
                continue

            df = pd.read_csv(csv_file)
            row = df[top_1_cols].iloc[0]
            values.append(row)

        if values:
            avg_df = pd.DataFrame(values).mean().to_frame("average").T
            avg_df.insert(0, "dataset", dataset)  # add dataset column
            all_results.append(avg_df)

    if all_results:
        result = pd.concat(all_results, ignore_index=True)
        result.to_csv(os.path.join(output_dir, "averages_top_1.csv"), index=False)
    else:
        print("No data found for top_1")


def compute_average_top_3():
    """Compute averages for top_3 using only TVSum with exclusions."""

    values = []
    csvs = collect_csvs("TVSum")
    for video_folder, csv_file in csvs:
        video_idx = get_video_index(video_folder)
        if video_idx in tvsum_excluded_top_3:
            continue
        df = pd.read_csv(csv_file)
        row = df[top_3_cols].iloc[0]
        values.append(row)

    if values:
        avg_df = pd.DataFrame(values).mean().to_frame("average").T
        avg_df.insert(0, "dataset", "TVSum")
        avg_df.to_csv(os.path.join(output_dir, "averages_top_3.csv"), index=False)
    else:
        print("No data found for top_3")


if __name__ == "__main__":
    compute_average_top_1()
    compute_average_top_3()
