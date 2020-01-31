from cx_Freeze import setup, Executable

setup(
    name='EC_analysis',
    version='1.0.0',
    options={'build_exe': {
        'packages': ['kivy',
                     'sklearn',
                     'tkfilebrowser',
                     'pandas',
                     'matplotlib',
                     'os',
                     'numpy',
                     'pathlib',
                     'configparser',
                     'scipy',
                     'kivy.garden.matplotlib.backend_kivyagg'
                     ],
        'include_files': ['interface.kv',
                          'analysis_button.png',
                          'background.png',
                          'config.ini',
                          'export_button.png',
                          'general_button.png',
                          'import_cv_button.png',
                          'ir_corr_button.png',
                          'navigation_button.png'
                          ],
        'include_msvcr': True
                            },
        'bdist_msi': {
            'all_users': False
                        }},
    executables=[Executable('interface.py')]
)
