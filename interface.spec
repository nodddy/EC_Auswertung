import kivy
from kivy.deps import sdl2, glew
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['interface.py'],
             pathex=['C:\\Users\\MaGoll\\PycharmProjects\\EC_Auswertung\\dist', '~/kivy/garden/matplotlib'],
             binaries=[],
             datas=[],
             hiddenimports=['kivy.deps.sdl2','pandas','tkfilebrowser', 'backend_kivy', 'kivy.garden', 'kivy', 'FileDialog', 'Tkinter', 'PIL._tkinter._finder', 'numpy.random.common', 'numpy.random.bounded_integers', 'numpy.random.entropy'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='interface',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
