import os
from git import Repo
from pathlib import Path

cwd = Path(os.getcwd())

if 'interface.py' in os.listdir(cwd):
    os.system("python interface.py")
else:
    Repo.clone_from('https://github.com/nodddy/EC_Auswertung', cwd / 'EC Analysis')
    os.rename(cwd / 'executable.exe', cwd / 'EC Analysis' / 'executable.exe')
    os.chdir(cwd / 'EC Analysis')
    os.system("python setup.py install")
    os.system("python interface.py")
