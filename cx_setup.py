from cx_Freeze import setup, Executable

setup(name='Run',
      executables=[Executable("interface.py")])