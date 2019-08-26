import os
from git import Repo
from pathlib import Path
import shutil

print('Updating...')
os.chdir('..')
cwd = Path(os.getcwd())
Repo.clone_from('https://github.com/nodddy/EC_Auswertung', cwd / 'temp')
os.rename(cwd / 'EC Analysis' / 'Update.exe', cwd / 'temp' / 'Update.exe')
shutil.rmtree(cwd / 'EC Analysis', ignore_errors=True)
os.remove(cwd / 'EC Analysis')
os.rename(cwd / 'temp', cwd / 'EC Analysis')
os.chdir(cwd / 'EC Analysis')
# os.system("python setup.py install")
