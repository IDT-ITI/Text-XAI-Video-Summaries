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
project_root = os.path.abspath(os.path.join(this_dir, ".."))
base_data_path = os.path.join(project_root, "data")
llava_path = os.path.abspath(os.path.join(project_root, "LLAVA"))

llava_script = os.path.join(llava_path, "text_explain.py")
llava_env = "llava"
llava_workdir = llava_path


def copy_results_back(src_root, dest_root):
    """
    Copies generated text_explanation folders (if present) from the working copy
    back to the main project data directory.

    Parameters:
        src_root (str): Root path of the source "data" directory (inside working_dir).
        dest_root (str): Root path of the destination "data" directory (main project).
    """

    if not os.path.exists(src_root):
        print(f"Source data directory does not exist: {src_root}")
        return

    copied = []
    for root, dirs, _ in os.walk(src_root):
        if "text_explanation" in dirs:
            src_folder = os.path.join(root, "text_explanation")
            rel_path = os.path.relpath(src_folder, src_root)
            dest_folder = os.path.join(dest_root, rel_path)

            # Ensure destination parent exists
            os.makedirs(os.path.dirname(dest_folder), exist_ok=True)

            # If a text_explanation already exists, remove it to fully sync
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)

            shutil.copytree(src_folder, dest_folder)
            copied.append((src_folder, dest_folder))

    if not copied:
        print(f"No text_explanation folders found in {src_root}")


def run_second_project(script_path, conda_env_name, working_dir, args=None):
    """
    Run a script in the given conda environment and working directory.

    Parameters:
        script_path (str): Path to the Python script to execute.
        conda_env_name (str): Name of the Conda environment to use.
        working_dir (str): Directory in which to execute the script (must exist).
        args (list[str], optional): Extra command-line arguments to pass to the script.
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
        script_path,
    ]

    if args:
        cmd.extend(args)

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
    # Forward everything after the script name to the subprocess
    extra_args = sys.argv[1:]

    if not extra_args:
        print("Usage: python text_explanation.py -d <paths> [more paths]")
        sys.exit(1)

    run_second_project(
        script_path=llava_script,
        conda_env_name=llava_env,
        working_dir=llava_workdir,
        args=extra_args
    )