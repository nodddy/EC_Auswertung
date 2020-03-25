import subprocess
import sys

git_package=f'git+https://github.com/nodddy/EC_Auswertung@master'
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install(git_package)