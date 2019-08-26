import os
from pathlib import Path

cwd = Path(os.getcwd())

if 'interface.py' in os.listdir(cwd):
    os.system("python interface.py")
else:
    clone_cmd = "git clone https://github.com/nodddy/EC_Auswertung"
    os.system(clone_cmd)
    os.rename(cwd / 'EC_Auswertung', cwd / 'EC Analysis')
    os.rename(cwd / 'AnalysisTool.exe', cwd / 'EC Analysis' / 'AnalysisTool.exe')
    os.chdir(cwd / 'EC Analysis')
    os.system("python setup.py install")
    os.system("garden install matplotlib")
    os.system("python interface.py")
