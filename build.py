import subprocess

make_process = subprocess.Popen(["make", "clean", "all"], stderr=subprocess.STDOUT)
if make_process.wait() != 0:
     raise Exception("Build failed")

