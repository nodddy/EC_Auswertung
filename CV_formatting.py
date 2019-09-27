from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

root_path = Path('N:/BZ_Neu/23_char_hp/5_Ergebnisse/MGo/Masterarbeit/Daten/RDE')


def format_csv(path):
    cv_list = [file for file in os.listdir(str(path)) if 'CV' in file and 'formatted' not in file]

    for file in cv_list:
        cv = pd.read_csv(path / file, sep='\t', skiprows=1)
        scan2 = cv.iloc[int(len(cv) / 3):int((len(cv) / 3) * 2)]
        scan3 = cv.iloc[int((len(cv) / 3) * 2):int(len(cv))]
        scan2.reset_index(inplace=True)
        scan3.reset_index(inplace=True)
        avg_scan = pd.DataFrame()
        avg_scan['WE(1).Current (A)'] = (scan2['WE(1).Current (A)'] + scan3['WE(1).Current (A)']) / 2
        avg_scan['WE(1).Potential (V)'] = (scan2['WE(1).Potential (V)'] + scan3['WE(1).Potential (V)']) / 2
        file_name = file.split(".")[-2]

        if os.path.isfile(str(path / file_name) + '_formatted.txt') is True:
            os.remove(str(path / file_name) + '_formatted.txt')
            print('deleted')

        with open(str(path / file_name) + '_formatted.txt', 'w') as new_file:
            new_file.write(pd.DataFrame.to_csv(avg_scan, sep='\t'))


for dir in os.listdir(str(root_path)):
    if os.path.isdir(str(root_path / dir)):
        format_csv(root_path / dir)
