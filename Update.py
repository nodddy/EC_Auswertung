import os
from pathlib import Path
import shutil

print('Updating...')

cwd = Path(os.getcwd())
clone_cmd = "git clone https://github.com/nodddy/EC_Auswertung"
os.system(clone_cmd)

for item in os.listdir(cwd/'EC_Auswertung'):
    try:
        shutil.move(str(cwd/'EC_Auswertung'/item), str(cwd/item))
    except shutil.Error:
        pass

os.system('rmdir /S /Q "{}"'.format(cwd/'EC_Auswertung'))

# os.system("python setup.py install")
