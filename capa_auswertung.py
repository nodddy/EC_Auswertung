import pandas as pd
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        self.data = None

        ''' 
        data structure: dict =
        {scan_rate: dataframe}
        '''

    def data(self):
        return self.data_before

    def set_data(self, data_before):
        self.data_before = data_before

    def import_data(self):
        def import_file(path):
            df = pd.read_csv(path, sep='\t')
            return df

        def format_data(imported_data):
            df_before = pd.DataFrame()
            df_after = pd.DataFrame()

            # hier rein den Namen der scan rate spalte
            # nur einzelnen wert nehmen
            scan_rate = imported_data['scan rate'].iloc[0]

            # hier rein wie die spalten beschriftung für vorher und nachher ist
            df_before['Pot'] = imported_data['Potential applied (V)(1)']
            df_before['Disk'] = imported_data['WE (1)']

            df_after['Pot'] = imported_data['Potential applied (V)(2)']
            df_after['Disk'] = imported_data['WE (2)']

            return scan_rate, {'before': df_before, 'after': df_after}

        folder = ''
        file_path = ''
        data_dict = {}

        for file in folder:
            file = import_file(file_path)
            scan_rate, dat = format_data(file)
            data_dict[str(scan_rate)] = dat

        return data_dict


class Auswertung:
    pass
