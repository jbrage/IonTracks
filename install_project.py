import subprocess
import sys
import os


def run_pip_command(command_parts, dependency_type, allow_failure=False):
    try:
        env = os.environ.copy()
        if "CYTHONIZE" in command_parts[-1]:
            env["CYTHONIZE"] = "1"
            command_parts = command_parts[:-1]

        subprocess.check_call(
            command_parts,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"{dependency_type} dependencies downloaded")
    except Exception as e:
        print(f"{dependency_type} dependencies not downloaded")
        if not allow_failure:
            print(e)
            sys.exit(1)



if __name__ == "__main__":
    run_pip_command(
        [sys.executable, "-m", "pip", "install", "--editable", ".", "CYTHONIZE=1"],
        "Basic"
    )

    run_pip_command(
        [sys.executable, "-m", "pip", "install", "cupy-cuda12x"],
        "GPU",
        True
    )