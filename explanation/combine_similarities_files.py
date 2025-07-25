import os
import pandas as pd

# Paths to the main folders
summe_folder = '../data/SumMe'
tvsum_folder = '../data/TVSum'


def find_csv_files(folder_path):
    """Recursively finds all *_similarities.csv files in subdirectories."""
    csv_files = []
    for root, _, files in os.walk(folder_path):  # Walk through all subfolders
        for file in files:
            if file.endswith("_similarities.csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def combine_selected_columns(folder_path, output_filename, selected_columns):
    """
    Combines selected columns from *_similarities.csv files in subdirectories of a given folder into a single CSV.
    Computes the average of each selected column.
    """
    csv_files = find_csv_files(folder_path)

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_selected = df[selected_columns].copy()  # Only keep selected columns
            df_selected["source_file"] = os.path.basename(file)
            dataframes.append(df_selected)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Compute averages
        avg_df = combined_df.drop(columns=["source_file"]).mean().to_frame().T

        # Save combined selected columns
        combined_output_path = os.path.join(folder_path, output_filename)
        combined_df.to_csv(combined_output_path, index=False)
        # print(f"Saved selected combined CSV to {combined_output_path}")

        # Save averages
        avg_output_path = os.path.join(folder_path, "Averages_" + output_filename)
        avg_df.to_csv(avg_output_path, index=False)
        # print(f"Saved selected column averages CSV to {avg_output_path}")


# Select which VideoSet to use
videoset = "VideoSet1"  # Change between "VideoSet1" for VideoSet1 and "VideoSet2" for VideoSet2

# Determine which columns to extract based on the VideoSet
if videoset == "VideoSet1":
    selected_columns = [
        'simcse_attention_sum', 'sbert_attention_sum',
        'simcse_lime_sum', 'sbert_lime_sum',
    ]
elif videoset == "VideoSet2":
    selected_columns = [
        'simcse_attention_sum', 'sbert_attention_sum',
        'simcse_lime_sum', 'sbert_lime_sum',
        'simcse_alternative_attention_sum', 'sbert_alternative_attention_sum',
        'simcse_alternative_lime_sum', 'sbert_alternative_lime_sum',
    ]
else:
    raise ValueError(f"Unknown videoset: {videoset}")

# Run for both datasets
combine_selected_columns(summe_folder, "SumMe_selected_combined.csv", selected_columns)
combine_selected_columns(tvsum_folder, "TVSum_selected_combined.csv", selected_columns)
