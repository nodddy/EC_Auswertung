from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.popup import Popup
from kivy.uix.recycleview import RecycleView

import tkfilebrowser
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import os
import numpy as np
from pathlib import Path
from configparser import ConfigParser
from scipy import integrate


class Config:
    """Interact with configuration variables."""

    configParser = ConfigParser()
    configFilePath = (os.path.join(os.getcwd(), 'config.ini'))

    @classmethod
    def initialize(cls):
        """Start config by reading config.ini."""
        cls.configParser.read(cls.configFilePath)

    @classmethod
    def glob(cls, key):
        """Get global values from config.ini."""
        return cls.configParser.get('GLOBAL', key)

    @classmethod
    def units(cls, key):
        """Get units values from config.ini."""
        return cls.configParser.get('UNITS', key)


class Data:
    def __init__(self):
        self.orr = None
        self.orr_bckg = None
        self.eis = None
        self.parameters = None
        self.anodic = None
        self.cathodic = None
        self.folder = None
        self.cv = None
        self.cv_eis = None

    def set_cv_eis(self, cv_eis):
        self.cv_eis = cv_eis

    def set_cv(self, cv):
        self.cv = cv

    def set_folder(self, folder):
        self.folder = folder.parents[0]

    def set_orr(self, dat):
        self.orr = dat

    def set_orr_bckg(self, dat):
        self.orr_bckg = dat

    def set_eis(self, dat):
        self.eis = dat

    def set_anodic(self, dat):
        self.anodic = dat

    def set_cathodic(self, dat):
        self.cathodic = dat

    def set_parameters(self, dat):
        self.parameters = dat

    def cv(self):
        return self.cv

    def cv_eis(self):
        return self.cv_eis

    def folder(self):
        return self.folder

    def orr(self):
        return self.orr

    def orr_bckg(self):
        return self.orr_bckg

    def eis(self):
        return self.eis

    def parameters(self):
        return self.parameters

    def anodic(self):
        return self.anodic

    def cathodic(self):
        return self.cathodic

    @staticmethod
    def reduce_data(dataset, window=3):
        smoothed = dataset.rolling(window=window).mean()
        return smoothed.iloc[::5].copy(deep=True).dropna()

    # data_descriptor is the object of the data stored in this class (data.orr, data.orr_bckg, data.eis)
    def import_data(self, data_descriptor):

        def import_from_path(row_skip=0):
            path = tkfilebrowser.askopenfilename(
                initialdir='N:/BZ_Neu/23_char_hp/5_Ergebnisse/MGo/Masterarbeit/Daten/RDE')
            file = pd.read_csv(path, sep='\t', skiprows=row_skip)
            data.set_folder(Path(path))
            return file, path

        def file_name_from_path(path):
            file_name = path.split(os.sep)[-1]
            return file_name.split('.')[0]

        def format_data(input, data_descriptor):

            def format_cv(input):
                scan2 = input.iloc[int(len(input) / 3):int((len(input) / 3) * 2)]
                scan3 = input.iloc[int((len(input) / 3) * 2):int(len(input))]
                scan2.reset_index(inplace=True)
                scan3.reset_index(inplace=True)
                avg_scan = pd.DataFrame()
                avg_scan['Disk'] = (((scan2['WE(1).Current (A)'] + scan3['WE(1).Current (A)']) / 2)
                                    * 1000) / float(Config.glob('ELECTRODE_AREA'))
                avg_scan['Pot'] = (scan2['WE(1).Potential (V)'] + scan3['WE(1).Potential (V)']) / 2
                return avg_scan

            def format_eis(input):
                data = pd.DataFrame()
                data['Real'] = input["Z' (\u03A9)"]
                data['Imag'] = input["-Z'' (\u03A9)"]
                return data

            def format_orr(input):
                scan1 = input.query('Scan == 2')
                scan1.reset_index(inplace=True)
                scan2 = input.query('Scan == 3')
                scan2.reset_index(inplace=True)
                input_data = pd.DataFrame()
                input_data['Pot'] = ((scan1['Potential applied (V)'] + scan2['Potential applied (V)']) / 2)
                input_data['Disk'] = (((scan1['WE(1).Current (A)'] + scan2[
                    'WE(1).Current (A)']) / 2) * 1000) / float(Config.glob('ELECTRODE_AREA'))
                try:
                    input_data['Ring'] = (((scan1['WE(2).Current (A)'] + scan2[
                        'WE(2).Current (A)']) / 2) * 1000) / float(Config.glob('ELECTRODE_AREA'))
                except KeyError:
                    pass

                half_scan_end = int(len(input_data.index) / 2)
                anodic_scan = input_data.iloc[40:half_scan_end - 40]
                cathodic_scan = input_data.iloc[half_scan_end + 40:-40]
                cathodic_scan = cathodic_scan[::-1]

                data.set_anodic(anodic_scan)
                data.set_cathodic(cathodic_scan)

                input_data.reset_index(inplace=True)
                return input_data

            if data_descriptor == 'eis':
                formatted_data = format_eis(input)
            elif data_descriptor == 'cv':
                formatted_data = format_cv(input)
            else:
                formatted_data = format_orr(input)
            return formatted_data

        if data_descriptor == 'cv':
            raw_data, path = import_from_path(row_skip=1)
        else:
            raw_data, path = import_from_path()
        file_name = file_name_from_path(path)
        if data_descriptor not in file_name.lower():
            raise FileNotFoundError
        formatted_data = format_data(raw_data, data_descriptor)
        return {'file_name': file_name, 'raw': raw_data, 'formatted': formatted_data, 'corrected': None, 'path': path}

    def correct_electrolyte_res(self, df_cv, df_eis):
        def find_electrolyte_res(df):
            electrolyte_res = df.iloc[(df['Imag']).abs().argsort()[:2]]
            return electrolyte_res['Real'].iloc[0]

        electrolyte_res = find_electrolyte_res(df_eis)
        df_cv['Pot'] = df_cv['Pot'] - (
                (df_cv['Disk'] / 1000) * electrolyte_res * float(Config.glob('ELECTRODE_AREA')))
        self.cv['corrected'] = df_cv
        return

    def correction(self, corr_dict):

        def correct_background(df, background_data):
            df['Disk'] = df['Disk'] - background_data['Disk']
            return df

        def correct_electrolyte_res(ORR_data, EIS_data):
            def find_electrolyte_res(df):
                electrolyte_res = df.iloc[(df['Imag']).abs().argsort()[:2]]
                return electrolyte_res['Real'].iloc[0]

            electrolyte_res = find_electrolyte_res(EIS_data)
            ORR_data['Pot'] = ORR_data['Pot'] - (
                    (ORR_data['Disk'] / 1000) * electrolyte_res * float(Config.glob('ELECTRODE_AREA')))
            return ORR_data

        def correct_electrode(df, kalibrierung=0, pH=1):
            df['Pot'] = df['Pot'] - kalibrierung
            return df

        self.orr['corrected'] = self.orr['formatted'].copy(deep=True)

        if corr_dict['electrode_corr'] is True:
            correct_electrode(self.orr['corrected'])
            if corr_dict['background_corr'] is True:
                self.orr_bckg['corrected'] = self.orr_bckg['formatted'].copy(deep=True)
                correct_electrode(self.orr_bckg['corrected'])

        if corr_dict['background_corr'] is True:
            self.orr_bckg['corrected'] = self.orr_bckg['formatted'].copy(deep=True)
            correct_background(self.orr['corrected'], self.orr_bckg['corrected'])

        if corr_dict['eis_corr'] is True:
            correct_electrolyte_res(self.orr['corrected'], self.eis['formatted'].copy(deep=True))

        half_scan_end = int(len(data.orr['corrected'].index) / 2)
        anodic_scan = data.orr['corrected'].iloc[60:half_scan_end - 60]
        cathodic_scan = data.orr['corrected'].iloc[half_scan_end + 60:-60]
        cathodic_scan = cathodic_scan[::-1]

        self.anodic = anodic_scan
        self.cathodic = cathodic_scan
        return


data = Data()


class Analysis:
    def __init__(self):
        self.cv = None
        self.orr = None

    def orr(self):
        return self.orr

    def cv(self):
        return self.cv

    def analyse_cv(self):
        pot_dict = {'hupd': (0.3, 0.6),
                    'co': ()}

        def find_ecsa(cv_data, technique='hupd'):
            def integrate_curve(curve_y, curve_x):
                return integrate.trapz(curve_y, curve_x)

            def find_curve_section(cv_data, lower_pot, upper_pot):
                min_point = cv_data.query('@lower_pot < Pot < @upper_pot', inplace=False).abs()
                min_point_curr = min_point['Disk'].min()
                min_point_pot = min_point.query('Disk == @min_point_curr')['Pot'].iloc[0]
                return cv_data.abs().query(' Pot <= @min_point_pot and Disk >= @min_point_curr',
                                           inplace=False)

            ecsa_curve = find_curve_section(cv_data, pot_dict[technique][0], pot_dict[technique][1])
            ecsa_area = integrate_curve(ecsa_curve['Disk'], ecsa_curve['Pot'])
            capa_area = integrate_curve(np.full(len(ecsa_curve), ecsa_curve['Disk'].min()),
                                        ecsa_curve['Pot'])
            if cv_data['Disk'].iloc[0] < 0:
                ecsa_curve['Disk'] = ecsa_curve['Disk'] * -1
            plt.fill(ecsa_curve['Pot'], ecsa_curve['Disk'])
            plotter.overwrite_plot()
            scan_rate = 0.05  # in V/s
            return (ecsa_area - capa_area) / (2100 * scan_rate * float(Config.glob('PT_LOADING')))

        try:
            cv_data = data.cv['corrected'].copy(deep=True)
            stage = 'corrected'
        except AttributeError:
            cv_data = data.cv['formatted'].copy(deep=True)
            stage = 'formatted'

        parameters_anod = {'ecsa': find_ecsa(cv_data.query('Disk > 0', inplace=False))}
        parameters_cath = {'ecsa': find_ecsa(cv_data.query('Disk < 0', inplace=False))}
        self.cv = [stage, parameters_anod, parameters_cath]
        return

    def analyse_orr(self):
        if data.orr is None:
            print('No ORR data found')
            return
        elif data.orr['corrected'] is None:
            orr = data.orr['formatted'].copy(deep=True)
            parameters = ['formatted']
        else:
            orr = data.orr['corrected'].copy(deep=True)
            parameters = ['corrected']

        reduced_anodic = data.reduce_data(data.anodic)
        reduced_anodic['Diff1'] = (reduced_anodic['Disk'].diff() / reduced_anodic['Pot'].diff()).dropna()
        reduced_cathodic = data.reduce_data(data.cathodic)
        reduced_cathodic['Diff1'] = (reduced_cathodic['Disk'].diff() / reduced_cathodic['Pot'].diff()).dropna()

        def find_inflectionpoint(input_data):
            diff_max = input_data['Diff1'].max()
            inflection_point = input_data.query('Diff1 == @diff_max', inplace=False)
            return inflection_point['Pot'].iloc[0]

        def find_onset(input_data):
            df = pd.DataFrame()
            curr_max = abs(input_data['Disk']).max()
            curr_min = abs(input_data['Disk']).min()
            df['Disk'] = (abs(input_data['Disk']) - curr_min) / (curr_max - curr_min)
            df['Pot'] = input_data['Pot']
            onset = df.query('Disk < 0.02', inplace=False)
            return onset['Pot'].iloc[0]

        def find_peroxide_yield(input_data, N):
            data = input_data.query('0.1 < Pot < 0.7', inplace=False).copy(deep=True)
            data['PeroxideYield'] = ((2 * (abs(data['Ring'] / N))) / (
                    abs(data['Disk']) + (abs(data['Ring']) / N))) * 100
            return data['PeroxideYield'].mean()

        def find_activity(input_data):
            def find_jlim(input_data):
                return -abs(input_data['Disk']).max()

            def find_fixed_potential(input_data, potential):
                current_df = input_data.iloc[(input_data['Pot'] - potential).abs().argsort()[:2]]
                current = current_df['Disk'].mean()
                return current

            j = find_fixed_potential(input_data, float(Config.glob('J_KIN_POTENTIAL')))
            j_lim = find_jlim(input_data)
            return (j * j_lim) / (j_lim - j)

        def find_Tafel(input):
            j_lim = -abs(input['Disk']).max()
            df = pd.DataFrame()
            df['Pot'] = input['Pot']
            df['Disk'] = input['Disk']
            df.query('0.4 < Pot', inplace=True)
            df['Tafel'] = (df['Disk'] * j_lim) / (j_lim - df['Disk']) * (-1)
            df['Tafel'] = np.log10(df['Tafel'])
            out_df = df.iloc[::20, :]
            return out_df

        def find_n(input_data, N):
            df = input_data.query('0.1 < Pot < 0.7', inplace=False).copy(deep=True)
            df['n'] = (4 * df['Disk']) / (df['Disk'] + (df['Ring'] / N))
            return df['n'].mean()

        for scan in [reduced_anodic, reduced_cathodic]:
            parameters.append({'halfwave_pot': find_inflectionpoint(scan),
                               'onset': find_onset(scan),
                               'peroxide_yield': find_peroxide_yield(scan, float(Config.glob('RRDE_N'))),
                               'n': find_n(scan, float(Config.glob('RRDE_N'))),
                               'activity': find_activity(scan),
                               'tafel': find_Tafel(scan)})
        self.orr = parameters
        return


analysis = Analysis()


class Plotter:
    current_plot = None

    plot_dict = {'raw ORR': {'data': [{'x': 'data.orr["formatted"]["Pot"]',
                                       'y': "data.orr['formatted']['Disk']"}],
                             'x_label': 'Pot vs. RHE [V]',
                             'y_label': 'Disk Current Density [mA/cm2]',
                             'title': 'raw ORR',
                             'clear_fig': False},
                 'ORR background': {'data': [{'x': 'data.orr_bckg["formatted"]["Pot"]',
                                              'y': "data.orr_bckg['formatted']['Disk']"}],
                                    'x_label': 'Pot vs. RHE [V]',
                                    'y_label': 'Disk Current Density [mA/cm2]',
                                    'title': 'ORR background',
                                    'clear_fig': True},
                 'EIS': {'data': [{'x': 'data.eis["formatted"]["Real"]',
                                   'y': 'data.eis["formatted"]["Imag"]'}],
                         'x_label': "Real Z' (\u03A9)",
                         'y_label': "Imaginary -Z'' (\u03A9)",
                         'title': 'EIS',
                         'clear_fig': True},
                 'ORR corrected': {'data': [{'x': 'data.orr["corrected"]["Pot"]',
                                             'y': 'data.orr["corrected"]["Disk"]'}],
                                   'x_label': 'Pot vs. RHE [V]',
                                   'y_label': 'Disk Current Density [mA/cm2]',
                                   'title': 'ORR corrected',
                                   'clear_fig': False},
                 'ORR analysis': {'data': [{'x': "data.anodic['Pot']",
                                            'y': "data.anodic['Disk']"},
                                           {'x': "data.cathodic['Pot']",
                                            'y': "data.cathodic['Disk']"}],
                                  'x_label': 'Pot vs. RHE [V]',
                                  'y_label': 'Disk Current Density [mA/cm2]',
                                  'title': 'ORR analysis',
                                  'clear_fig': True,
                                  'parameters': True},
                 'Tafel plot': {'data': [{'x': "analysis.orr[2]['tafel']['Pot']",
                                          'y': "analysis.orr[2]['tafel']['Tafel']"}],
                                'y_label': 'log(jkin)',
                                'x_label': 'Potential vs. RHE [V]',
                                'title': 'Tafel plot',
                                'clear_fig': True},
                 'raw CV': {'data': [{'x': 'data.cv["formatted"]["Pot"]',
                                      'y': "data.cv['formatted']['Disk']"}],
                            'y_label': 'Current Density [mA/cm2]',
                            'x_label': 'Potential vs. RHE [V]',
                            'title': 'raw CV',
                            'clear_fig': True,
                            'zero_line': True},
                 'corrected CV': {'data': [{'x': 'data.cv["corrected"]["Pot"]',
                                            'y': "data.cv['corrected']['Disk']"}],
                                  'y_label': 'Current Density [mA/cm2]',
                                  'x_label': 'Potential vs. RHE [V]',
                                  'title': 'corrected CV',
                                  'clear_fig': False,
                                  'zero_line': True}}

    @staticmethod
    def overwrite_plot():
        active_screen = manager.get_screen(manager.current)
        active_screen.ids['plotter'].clear_widgets()
        active_screen.ids['plotter'].add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def plot(self, plot_name):
        if plot_name not in self.plot_dict.keys():
            return

        if self.plot_dict[plot_name]['clear_fig'] is True:
            plt.cla()
        if self.current_plot is not None:
            if self.plot_dict[self.current_plot]['clear_fig'] is True:
                plt.cla()
        self.current_plot = plot_name

        plt.xlabel(self.plot_dict[plot_name]['x_label'])
        plt.ylabel(self.plot_dict[plot_name]['y_label'])
        plt.title(self.plot_dict[plot_name]['title'], pad=25, fontdict={'fontsize': 18})
        for dataset in self.plot_dict[plot_name]['data']:
            plt.plot(eval(dataset['x']), eval(dataset['y']))
        if 'parameters' in self.plot_dict[plot_name].keys():
            self.plot_parameters()
        if 'zero_line' in self.plot_dict[plot_name].keys():
            x_data = eval(self.plot_dict[plot_name]['data'][0]['x'])
            plt.plot(x_data, np.full((len(x_data)), 0), color='r', linestyle='dashed')

        return self.overwrite_plot()

    def plot_parameters(self):

        def get_points(pot_value, dataset):
            return dataset.iloc[(dataset['Pot'] - pot_value).abs().argsort()[:1]]

        anod_list = [analysis.orr[1]['halfwave_pot'], analysis.orr[1]['onset']]
        cath_list = [analysis.orr[2]['halfwave_pot'], analysis.orr[2]['onset']]

        for item in anod_list:
            row = get_points(item, data.anodic)
            plt.scatter(row['Pot'], row['Disk'])
        for item in cath_list:
            row = get_points(item, data.cathodic)
            plt.scatter(row['Pot'], row['Disk'])
        return


class ScreenOne(Screen):

    def __init__(self, **kwargs):
        super(ScreenOne, self).__init__(**kwargs)
        Window.clearcolor = (0.5, 0.5, 0.5, 1)
        Window.borderless = False
        self.active_RV = None

    def add_plot_button(self, btn_text):
        self.ids['plot_spinner'].values.append(btn_text)
        self.ids['plot_spinner'].values = list(dict.fromkeys(self.ids['plot_spinner'].values))

    @staticmethod
    def plot_from_spinner(value):
        plotter.plot(value)

    def clear_plot_spinner(self):
        self.ids['plot_spinner'].values = []

    @staticmethod
    def clear_plot():
        plt.cla()
        plt.xlabel('Potential [V]')
        plt.ylabel('Current[A]')
        return plotter.overwrite_plot()

    @staticmethod
    def safe_plot_to_png():
        if data.orr is None:
            return

        path = Path(tkfilebrowser.asksaveasfilename(initialdir=data.folder) + '.png')
        if path == '.png':
            return
        return plt.savefig(fname=str(path), format='png')

    def active_data_RV(self):
        widget = RV(data.orr, data.orr_bckg, data.eis)
        self.ids['messages'].add_widget(widget)
        self.active_RV = widget
        return

    def import_orr_single(self):
        try:
            data.set_orr(data.import_data('orr'))
        except FileNotFoundError:
            return
        if self.active_RV is not None:
            RV.on_update(self.active_RV)
        self.active_data_RV()
        self.clear_plot_spinner()
        self.add_plot_button('raw ORR')
        plotter.plot('raw ORR')
        return

    def import_orr_bckg(self):
        try:
            data.set_orr_bckg(data.import_data('orr'))
        except FileNotFoundError:
            print('No ORR data found')
            return
        if self.active_RV is not None:
            RV.on_update(self.active_RV)
        self.active_data_RV()
        self.add_plot_button('ORR background')
        return

    def import_eis(self):
        try:
            data.set_eis(data.import_data('eis'))
        except FileNotFoundError:
            print('No EIS data found')
            return
        if self.active_RV is not None:
            RV.on_update(self.active_RV)
        self.active_data_RV()
        self.add_plot_button('EIS')
        return

    def open_correction_popup(self):
        if data.orr is not None:
            popup = DataCorrection(data.orr_bckg, data.eis)
            return popup.open()

    def do_correct_data(self, corr_dict):
        return self.correct_data(corr_dict)

    def correct_data(self, corr_dict):
        data.correction(corr_dict)
        self.add_plot_button('ORR corrected')
        plotter.plot('ORR corrected')
        return

    def analyse_data(self):
        analysis.analyse_orr()
        self.show_parameters()
        self.add_plot_button('ORR analysis')
        self.add_plot_button('Tafel plot')
        plotter.plot('ORR analysis')
        return

    @staticmethod
    def show_parameters(shown_parameters='analysis.orr', top=0.6, center_x=0.82):
        root = manager.get_screen(manager.current).ids['parameters']
        root.pos_hint = {"top": top, "center_x": center_x}
        root.children[1].clear_widgets()
        root.children[1].add_widget(Label(text='anodic', bold=True, color=(0, 0, 0, 1)))
        root.children[2].clear_widgets()
        root.children[2].add_widget(Label(text='cathodic', bold=True, color=(0, 0, 0, 1)))
        params = eval(shown_parameters)
        for i in range(1, 3):
            for key in params[i]:
                if key is not 'tafel':
                    val = params[i][key].round(3)
                    lbl = Label(text=str(val), color=(0, 0, 0, 1), font_size= 20)
                    root.children[i].add_widget(lbl)

    # Export corrected data and summary to txt and csv
    # Export all plots to png
    def export_data(self, data_to_export='orr'):

        def write_to_file(dict, file_path):
            with open(file_path, 'w') as file:
                file.write('Parameter \t Value \n' + csv_from_dict(dict))

        def csv_from_dict(dict):
            csv = ''
            for k, v in dict.items():
                if isinstance(v, type(dict)):
                    line = k + '\n' + csv_from_dict(v)
                    csv += line
                    continue
                line = str(k) + '\t' + str(v) + '\n'
                csv += line
            return csv

        def rename_header(df, dict):
            return df.rename(columns=dict, inplace=True)

        if data.orr is None:
            return

        dir = Path(tkfilebrowser.askopendirname(initialdir=data.folder))
        if dir == Path('.'):
            return
        corr_orr_file_name = data.orr['file_name'] + '_corrected.txt'
        anodic_file_name = data.orr['file_name'] + '_anodic.txt'
        cathodic_file_name = data.orr['file_name'] + '_cathodic.txt'
        parameters_ano_file_name = data.orr['file_name'] + '_parameters_anodic.txt'
        parameters_cat_file_name = data.orr['file_name'] + '_parameters_cathodic.txt'

        orr_header_dict = {'Pot': 'Potential [V]',
                           'Disk': 'Disk Current density[mA/cm2]',
                           'Ring': 'Ring Current density[mA/cm2]'}

        for item in [data.anodic, data.cathodic, data.orr['corrected']]:
            try:
                rename_header(item, orr_header_dict)
            except AttributeError:
                pass
        try:
            data.orr['corrected'].to_csv(dir / corr_orr_file_name, sep='\t', index=False, header=True)
        except AttributeError:
            pass
        data.anodic.to_csv(dir / anodic_file_name, sep='\t', index=False, header=True)
        data.cathodic.to_csv(dir / cathodic_file_name, sep='\t', index=False, header=True)
        try:
            write_to_file(analysis.orr[1], dir / parameters_ano_file_name)
            write_to_file(analysis.orr[2], dir / parameters_cat_file_name)
        except TypeError:
            return


class ScreenTwo(Screen):

    def import_cv(self):
        try:
            data.set_cv(data.import_data('cv'))
        except FileNotFoundError:
            return
        plotter.plot('raw CV')
        return

    def analyse(self):
        if data.cv is None:
            return
        analysis.analyse_cv()
        ScreenOne.show_parameters(shown_parameters='analysis.cv',
                                  center_x=0.4)
        return

    def clear_plot(self):
        plt.cla()
        plt.xlabel('Potential [V]')
        plt.ylabel('Current[A]')
        return plotter.overwrite_plot()

    def safe_plot_to_png(self):
        if data.orr is None:
            return

        path = Path(tkfilebrowser.asksaveasfilename(initialdir=data.folder) + '.png')
        if path == '.png':
            return
        return plt.savefig(fname=str(path), format='png')

    def ir_correction(self):
        if data.cv is None:
            return
        try:
            data.set_cv_eis(data.import_data('eis'))
        except FileNotFoundError:
            print('No EIS data found')
        data.correct_electrolyte_res(data.cv['formatted'], data.cv_eis['formatted'])
        return plotter.plot('corrected CV')


class ScreenThree(Screen):
    pass


class RV(RecycleView):
    def __init__(self, orr_data, orr_bckg_data, eis_data, **kwargs):
        super(RV, self).__init__(**kwargs)
        data_list = []
        if orr_data is not None:
            data_list.append({'text': 'ORR: \n' + orr_data['file_name'], 'color': (0, 0, 0, 1), 'multiline': True})
        if orr_bckg_data is not None:
            data_list.append({'text': 'ORR background: \n' + orr_bckg_data['file_name'], 'color': (0, 0, 0, 1)})
        if eis_data is not None:
            data_list.append({'text': 'EIS: \n' + eis_data['file_name'], 'color': (0, 0, 0, 1)})
        self.data = data_list
        return

    def on_update(self):
        return self.parent.remove_widget(self)


class Navigation(Popup):
    def __init__(self, screen_manager, **kwargs):
        super(Navigation, self).__init__(**kwargs)
        self.manager = screen_manager
        return


class DataCorrection(Popup):
    def __init__(self, orr_bckg_data, eis_data, **kwargs):
        super(DataCorrection, self).__init__(**kwargs)

        if orr_bckg_data is None:
            lbl = Label(text='not available')
            self.orr_bckg_present = False
        else:
            lbl = Label(text='available')
            self.orr_bckg_present = True
        self.ids['background_corr'].add_widget(lbl)

        if eis_data is None:
            lbl = Label(text='not available')
            self.eis_present = False
        else:
            lbl = Label(text='available')
            self.eis_present = True
        self.ids['eis_corr'].add_widget(lbl)

    def on_activation(self, checkbox_obj):
        if checkbox_obj.parent.name == 'background_corr':
            if self.orr_bckg_present is False:
                checkbox_obj.active = False
        if checkbox_obj.parent.name == 'eis_corr':
            if self.eis_present is False:
                checkbox_obj.active = False

    def on_corr_confirmation(self):
        if self.ids['electrode'].active is True:
            self.electrode_corr = True
        else:
            self.electrode_corr = False

        if self.ids['bckg'].active is True:
            self.bckg_corr = True
        else:
            self.bckg_corr = False

        if self.ids['eis'].active is True:
            self.eis_corr = True
        else:
            self.eis_corr = False

        self.dismiss()
        corr_dict = {'electrode_corr': self.electrode_corr, 'background_corr': self.bckg_corr,
                     'eis_corr': self.eis_corr}
        return ScreenOne.do_correct_data(manager.ids['screen_one'], corr_dict)


plotter = Plotter()


class Manager(ScreenManager):
    screen_one = ObjectProperty(None)
    screen_two = ObjectProperty(None)


class InterfaceApp(App):
    Window.maximize()
    Config.initialize()

    def build(self):
        global manager
        manager = Manager()
        return manager

    @staticmethod
    def open_navigation():
        navigation = Navigation(manager)
        return navigation.open()


if __name__ == '__main__':
    InterfaceApp().run()
