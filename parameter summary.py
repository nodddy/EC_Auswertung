from pathlib import Path
import pandas as pd
import os

dir = Path('N:/BZ_Neu/23_char_hp/5_Ergebnisse/MGo/Masterarbeit/Daten/RDE')


def import_data(dir):
    def format_from_csv():
        df = pd.read_csv(dir / folder / 'analysis' / file, sep='\t').transpose()
        df.columns = df.iloc[0]
        df.drop(df.index[0], inplace=True)
        return df

    dict = {}
    for folder in os.listdir(str(dir)):
        try:
            date = folder.split('_')[0]
            before = None
            after = None
            for file in os.listdir(dir / folder / 'analysis'):
                if 'parameters' in file:
                    mode = str(file.split('_')[-1]).split('.')[0]
                    if 'before' in file:
                        before = format_from_csv()
                    if 'after' in file:
                        after = format_from_csv()
                    mode_dict = {str(mode): {'folder': str(Path(dir / folder)), 'before': before, 'after': after}}
                    try:
                        dict[date].update(mode_dict)
                    except KeyError:
                        dict[date] = mode_dict
        except FileNotFoundError:
            continue
    return dict


def analyse_data(data, parameters):
    loss_dict = {}

    for date in data.keys():
        loss_dict[date] = {}
        for mode in ['anodic', 'cathodic']:
            if data[date][mode]['before'] is None or data[date][mode]['after'] is None:
                loss_dict.pop(str(date), None)
                continue

            loss_dict[date][mode] = {}
            for param in parameters:
                value_before = data[date][mode]['before'][str(param)].iloc[0]
                value_after = data[date][mode]['after'][str(param)].iloc[0]
                loss = (1 - (value_after / value_before))*100
                loss_dict[date][mode][str(param)] = loss
                loss_dict[date][mode][str(param) + '_value_before'] = value_before
                loss_dict[date][mode][str(param) + '_value_after'] = value_after
    return loss_dict


def export_summary(data_dict):
    if os.path.isfile(dir / 'parameter summary.txt') is True:
        os.remove(dir / 'parameter summary.txt')
        print('deleted')

    for key in data_dict.keys():
        with open(dir / 'parameter summary.txt', 'a+') as file:
            file.write(str(key) + '\n' + pd.DataFrame.to_csv(pd.DataFrame.from_dict(data_dict[key]), sep='\t'))


data = import_data(dir)
export_summary(analyse_data(data, ['peroxide_yield', 'activity', 'halfwave_pot', 'onset', 'n']))
