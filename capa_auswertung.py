import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
import numpy as np


class Data:
    def __init__(self):
        self.data = None

        ''' 
        data structure: dict =
        {scan_rate: dataframe}
        '''

    def data(self):
        return self.data

    def set_data(self, data_dict):
        self.data = data_dict

    def import_new_data(self, path):
        def import_file(file_path):
            df = pd.read_csv(file_path, sep='\t')
            return df

        def format_data(df):
            split_df_list = []
            df.rename(columns={'Column 1 (V/s)': 'scan_rate',
                               'Potential applied (V)': 'Pot',
                               'WE(1).Current (A)': 'Disk'},
                      inplace=True)
            new_df = df[~df['scan_rate'].str.contains("Column", na=False)]
            new_df.reset_index(inplace=True)
            index_lst = list(new_df.query('scan_rate == scan_rate').index)
            index_lst.remove(0)
            index = 0
            for item in index_lst:
                split_df = new_df.iloc[index: item - 2]
                split_df_list.append(split_df)
                index = item
            key_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
            return dict(zip(key_list, split_df_list))

        for file in os.listdir(path):
            if 'hin' in file and 'before' in file:
                return self.set_data(format_data(import_file(path / file)))
        return self.set_data(None)

    def import_data(self, path):
        def import_file(path):
            df = pd.read_csv(path, sep='\t')
            return df

        def format_data(imported_data):
            df = pd.DataFrame()

            scan_rate = imported_data['Column 1 (V/s)'].iloc[0]

            df['Pot'] = imported_data['Potential applied (V)']
            df['Disk'] = imported_data['WE(1).Current (A)']

            return scan_rate, df

        data_dict = {}

        for file in os.listdir(path / 'capa'):
            if 'hin' in file and 'after' in file:
                raw_import = import_file(path / file)
                scan_rate, formatted_data = format_data(raw_import)
                data_dict[str(scan_rate)] = formatted_data

        return data_dict


class Auswertung:
    analysis_potential = 0.35

    def get_currents(self):
        """ gets current values at fixed potential from class (analysis_potential) """

        if data.data is None:
            return None

        values = []
        for scan_rate in data.data.keys():
            current = data.data[scan_rate].iloc[
                (data.data[scan_rate]['Pot'].astype(float) - self.analysis_potential).abs().argsort()[:2]][
                'Disk'].astype(float).iloc[0]
            values.append((scan_rate, current))
        return values

    def linear_regr(self, value_lst):
        """ applies linear reggression to x and y values of tuple list and return slope value """

        if value_lst is None:
            return None
        x_lst = [tpl[0] for tpl in value_lst]
        y_lst = [tpl[1] for tpl in value_lst]
        model = LinearRegression().fit(np.asarray(x_lst).astype(np.float64).reshape(-1, 1),
                                       np.asarray(y_lst).astype(np.float64))
        return model.coef_[0]/0.2472, model.intercept_/0.2472


class Plotter:

    def plot(self):
        for key in data.data.keys():
            plt.plot(data.data[key]['Pot'].astype(float), data.data[key]['Disk'].astype(float))

        plt.show()


data = Data()
auswertung = Auswertung()
plotter = Plotter()


def analyse_capacity(path):
    if os.path.isdir(path) is True:
        data.import_new_data(path)
        val = auswertung.get_currents()
        try:
            slope, intercept = auswertung.linear_regr(val)
        except TypeError:
            slope, intercept = None, None
        return str(path).split('20')[-1], slope, intercept
    else:
        return None,None,None

def csv_from_dict(dict):
    csv = ''
    for k, v in dict.items():
        if k == 'data':
            continue
        if isinstance(v, type(dict)):
            line = k + '\n' + csv_from_dict(v)
            csv += line
            continue
        line = str(k) + '\t' + str(v) + '\n'
        csv += line
    return csv

dir = Path('N:/BZ_Neu/23_char_hp/5_Ergebnisse/MGo/Masterarbeit/Daten/RDE')
dicti = {}
for folder in os.listdir(dir):
    key, val1, val2 = analyse_capacity(dir / folder)
    if val1 is not None:
        dicti[key] = f'{val1}\t{val2}'

with open(dir/'capa_analysis.txt', 'w') as file:
    file.write(f'Parameter \t Value \n{csv_from_dict(dicti)}')


