from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

dir = Path('N:/BZ_Neu/23_char_hp/5_Ergebnisse/MGo/Masterarbeit/Daten/RDE')

cv_files = {}
for folder in os.listdir(str(dir)):
    if not os.path.isdir(dir / folder):
        continue

    before = None
    after = None
    for file in os.listdir(str(dir / folder)):
        if not os.path.isfile(dir / folder / file):
            continue

        if 'CV' in file and 'formatted' in file:
            if 'before' in file:
                before = pd.read_csv(dir / folder / file, sep='\t')
            elif 'after' in file:
                after = pd.read_csv(dir / folder / file, sep='\t')
    cv_files[folder] = (before, after)

loss_values = {}
for key in cv_files:
    if cv_files[key][0] is None or cv_files[key][1] is None:
        continue
    after_val = cv_files[key][1]['WE(1).Current (A)'].max()
    pot_max = cv_files[key][1][cv_files[key][1]['WE(1).Current (A)'] == after_val]['WE(1).Potential (V)'].iloc[0]
    before_val = cv_files[key][0].iloc[(cv_files[key][0]['WE(1).Potential (V)'] - pot_max).abs().argsort()[:2]][
        'WE(1).Current (A)'].iloc[0]

    loss_values[key] = ((after_val / before_val) * 100 - 100)

print(loss_values)
