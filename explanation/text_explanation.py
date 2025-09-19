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

def run_second_project(script_path, conda_env_name, working_dir, args=None):
    """
    Run a script in the given environment and working directory.
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
        print("Finished run.")



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