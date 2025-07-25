import os
import subprocess
import warnings
import sys
import shutil
sys.path.append('../')

warnings.simplefilter(action='ignore', category=FutureWarning)

# Get the absolute path to the directory this script is in
this_dir = os.path.dirname(os.path.abspath(__file__))

# Paths relative to this script
project_root = os.path.abspath(os.path.join(this_dir, ".."))  # LLAVA-XAI_copy
base_data_path = os.path.join(project_root, "data")
llava_path = os.path.abspath(os.path.join(project_root, "LLAVA"))

datasets = ["TVSum", "SumMe"]

##### Script "evaluate2.py" runs for Video Set 2 (At least 3 top scoring fragments).
##### Change script name to "evaluate.py" for Video Set 1 (At least 1 top scoring fragment)
llava_script = os.path.join(llava_path, "evaluate2.py")
llava_env = "llava"
llava_workdir = llava_path

def generate_txt_files(video_path, dataset):
    """
    Parses explanation and shot segmentation files in a video folder to generate formatted text files
    needed for llava text generation and evaluation script.

    Generated files include:
        - {video_id}_sum_shots.txt: Sorted (temporal order) Top Fragments section.
        - {video_id}_attention_importance.txt: Raw Attention section.
        - {video_id}_attention_explanations.txt: Sorted (temporal order) Attention section.
        - {video_id}_lime_importance.txt: Raw LIME section.
        - {video_id}_lime_explanations.txt: Sorted (temporal order) LIME section.

    Parameters:
        video_path (str): Path to the video folder containing explanation and shots files.
        dataset (str): Name of the dataset the video belongs to (currently unused in this function).
    """

    explanation_file = os.path.join(video_path, "explanation", "explanation_and_top_fragments.txt")
    # Determine shots file (check renamed file first)
    video_id = os.path.basename(video_path)
    renamed_shots = os.path.join(video_path, f"{video_id}_shots.txt")

    if os.path.exists(renamed_shots):
        shots_file = renamed_shots
    else:
        # Look for original names
        shots_file = next(
            (os.path.join(video_path, f) for f in ["shots.txt", "opt_shots.txt"] if
             os.path.exists(os.path.join(video_path, f))),
            None
        )
        if shots_file:
            os.rename(shots_file, renamed_shots)
            shots_file = renamed_shots

    if not os.path.exists(explanation_file) or shots_file is None:
        print(f"Missing explanation or shots file for {video_path}")
        return

    # Rename the original shots file to {video_id}_shots.txt
    video_id = os.path.basename(video_path)
    new_shots_file = os.path.join(video_path, f"{video_id}_shots.txt")
    if shots_file != new_shots_file:  # Avoid renaming if it's already named properly
        os.rename(shots_file, new_shots_file)


    with open(explanation_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    sections = {"Attention": [], "LIME": [], "Top Fragments": []}
    current = None

    for line in lines:
        key = line.rstrip(":")
        if key in sections:
            current = key
        elif current:
            sections[current].append(line)

    def sort_fragments(fragments):
        return sorted(fragments, key=lambda x: int(x.split(",")[0]))

    attention = sections["Attention"]
    lime = sections["LIME"]
    top = sections["Top Fragments"]

    video_id = os.path.basename(video_path)

    out_files = {
        f"{video_id}_sum_shots.txt": sort_fragments(top),
        f"{video_id}_attention_importance.txt": attention,
        f"{video_id}_attention_explanations.txt": sort_fragments(attention),
        f"{video_id}_lime_importance.txt": lime,
        f"{video_id}_lime_explanations.txt": sort_fragments(lime)
    }

    for filename, content in out_files.items():
        target_path = os.path.join(video_path, filename)
        with open(target_path, "w") as f:
            f.write("\n".join([x.strip() for x in content]) + "\n")

def prepare_all_videos():
    """
    Processes all videos across datasets by generating explanation-related text files.
    """
    for dataset in datasets:
        dataset_path = os.path.join(base_data_path, dataset)
        if not os.path.exists(dataset_path):
            continue

        for video_folder in os.listdir(dataset_path):
            video_path = os.path.join(dataset_path, video_folder)
            if not os.path.isdir(video_path):
                continue
            generate_txt_files(video_path, dataset)

def copy_results_back(src_root, dest_root):
    """
    Copies generated text explanation and similarities scores from llava working directory
    back to each video subfolder.

    Parameters:
        src_root (str): Root path of the source directory containing datasets.
        dest_root (str): Root path of the destination directory to copy results to.
    """
    for dataset in datasets:
        src_dataset_path = os.path.join(src_root, dataset)
        dest_dataset_path = os.path.join(dest_root, dataset)

        if not os.path.exists(src_dataset_path):
            continue

        for video_folder in os.listdir(src_dataset_path):
            src_video_path = os.path.join(src_dataset_path, video_folder)
            dest_video_path = os.path.join(dest_dataset_path, video_folder)

            if not os.path.isdir(src_video_path) or not os.path.isdir(dest_video_path):
                continue

            for file in os.listdir(src_video_path):
                if file.endswith("text.txt") or file.endswith("_similarities.csv"):
                    src_file = os.path.join(src_video_path, file)
                    dest_file = os.path.join(dest_video_path, file)
                    shutil.copy2(src_file, dest_file)

def run_second_project(script_path, conda_env_name, working_dir):
    """
    Used to run llava text generation and evaluation script in a separate project environment with
    the corresponding conda environment.

    This function:
        - Copies the main `base_data_path` data directory into the `working_dir` under a new "data" folder.
        - After execution, copies any updated result files back to `base_data_path`.
        - Cleans up the temporary copied data directory.

    Parameters:
        script_path (str): Path to the Python script to execute.
        conda_env_name (str): Name of the Conda environment to use.
        working_dir (str): Directory in which to execute the script (must exist).
    """

    conda_path = os.environ.get("CONDA_EXE")
    if conda_path is None:
        raise EnvironmentError("Cannot locate conda. Try set CONDA_EXE.")

    cmd = [
        conda_path,
        "run",
        "-n",
        conda_env_name,
        "python",
        "-u",  # unbuffered output
        script_path
    ]

    # Copy data folder to working_dir
    src_data_path = base_data_path
    dest_data_path = os.path.join(working_dir, "data")

    if os.path.exists(dest_data_path):
        shutil.rmtree(dest_data_path)  # clean up if already exists
    shutil.copytree(src_data_path, dest_data_path)

    try:
        print(f"\nRunning: {' '.join(cmd)} in {working_dir}")

        process = subprocess.Popen(
            cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    finally:
        # Copy results back to main project before deleting
        copy_results_back(dest_data_path, base_data_path)

        # Cleanup
        if os.path.exists(dest_data_path):
            shutil.rmtree(dest_data_path)

if __name__ == "__main__":
    prepare_all_videos()
    run_second_project(
        script_path=llava_script,
        conda_env_name=llava_env,
        working_dir=llava_workdir
    )
