from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

dir = Path('N:/BZ_Neu/23_char_hp/5_Ergebnisse/MGo/Masterarbeit/Daten/RDE')

cv_files = {}
for folder in os.listdir(str(dir)):
    if not os.path.isdir(dir/folder):
        continue

    before = None
    after = None
    for file in os.listdir(str(dir / folder)):
        if not os.path.isfile(dir/folder/file):
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

    before_df_sorted = cv_files[key][0].iloc[(cv_files[key][0]['WE(1).Potential (V)'] - 0.7).abs().argsort()[:2]]
    after_df_sorted = cv_files[key][1].iloc[(cv_files[key][1]['WE(1).Potential (V)'] - 0.7).abs().argsort()[:2]]

    before_val=before_df_sorted['WE(1).Current (A)'].iloc[0]
    after_val = after_df_sorted['WE(1).Current (A)'].iloc[0]
    loss_values[key] = (before_val, after_val, (after_val/before_val)*100-100)

print(loss_values)
