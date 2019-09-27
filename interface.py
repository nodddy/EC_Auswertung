from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.label import Label
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

        def import_from_path():
            path = tkfilebrowser.askopenfilename(
                initialdir='N:/BZ_Neu/23_char_hp/5_Ergebnisse/MGo/Masterarbeit/Daten/RDE')
            file = pd.read_csv(path, sep='\t')
            data.set_folder(Path(path))
            return file, path

        def file_name_from_path(path):
            file_name = path.split(os.sep)[-1]
            return file_name.split('.')[0]

        def format_data(input, data_descriptor):

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
            else:
                formatted_data = format_orr(input)
            return formatted_data

        raw_data, path = import_from_path()
        file_name = file_name_from_path(path)
        if data_descriptor not in file_name.lower():
            raise FileNotFoundError
        formatted_data = format_data(raw_data, data_descriptor)
        return {'file_name': file_name, 'raw': raw_data, 'formatted': formatted_data, 'corrected': None, 'path': path}


class Plotter:
    active_screen = None
    plotter_widget = None

    @classmethod
    def set_active_screen(cls, screen):
        cls.active_screen = screen

    def overwrite_plot(self):
        self.active_screen.ids['plotter'].clear_widgets()
        self.active_screen.ids['plotter'].add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def plot_import_orr(self, data):
        plt.cla()
        plt.plot(data['formatted']['Pot'], data['formatted']['Disk'])
        plt.xlabel('Potential [mV]')
        plt.ylabel('Current density [mA/cm2]')
        return self.overwrite_plot()

    def plot_corr_orr(self):
        dat = data.orr
        plt.plot(dat['corrected']['Pot'], dat['corrected']['Disk'])
        plt.xlabel('Potential [V]')
        plt.ylabel('Current density [mA/cm2]')
        return self.overwrite_plot()

    def plot_parameters(self, anodic_scan, cathodic_scan):
        plt.cla()
        anod = data.anodic
        cath = data.cathodic
        plt.plot(cath['Pot'], cath['Disk'])
        plt.plot(anod['Pot'], anod['Disk'])
        param = data.parameters
        halfwave_anod = param[1]['halfwave_pot']
        halfwave_cath = param[2]['halfwave_pot']
        hw_point_anod = anodic_scan.query('Pot == @halfwave_anod')
        hw_point_cath = cathodic_scan.query('Pot == @halfwave_cath')
        plt.scatter(hw_point_anod['Pot'], hw_point_anod['Disk'])
        plt.scatter(hw_point_cath['Pot'], hw_point_cath['Disk'])

        onset_anod = param[1]['onset']
        onset_cath = param[2]['onset']
        onset_point_anod = anodic_scan.query('Pot == @onset_anod')
        onset_point_cath = cathodic_scan.query('Pot == @onset_cath')
        plt.scatter(onset_point_anod['Pot'], onset_point_anod['Disk'])
        plt.scatter(onset_point_cath['Pot'], onset_point_cath['Disk'])

        plt.xlabel('Potential [V]')
        plt.ylabel('Current density [mA/cm2]')

        # param[1]['tafel'] = param[1]['tafel'].rolling(window=30).mean()
        # plt.scatter(param[1]['tafel']['Pot'], param[1]['tafel']['Tafel'])
        return self.overwrite_plot()


class ScreenOne(Screen):

    def __init__(self, **kwargs):
        super(ScreenOne, self).__init__(**kwargs)
        Window.clearcolor = (1, 1, 1, 1)
        Window.borderless = False
        self.active_RV = None
        Plotter.set_active_screen(self)

    def clear_plot(self):
        plt.cla()
        plt.xlabel('Potential [V]')
        plt.ylabel('Current[A]')
        return plotter.overwrite_plot()

    def safe_plot_to_png(self):
        if data.orr is None:
            return

        dir = Path(tkfilebrowser.asksaveasfilename(initialdir=data.folder) + '.png')
        if dir == '.png':
            return
        plt.savefig(fname=str(dir), format='png')

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
        plotter.plot_import_orr(data.orr)
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
        return

    def open_correction_popup(self):
        if data.orr is not None:
            popup = DataCorrection(data.orr_bckg, data.eis)
            return popup.open()

    def do_correct_data(self, corr_dict):
        return self.correct_data(corr_dict)

    # RHE correction
    # IR correction
    # Background correction
    def correct_data(self, corr_dict):

        def correct_background(data, background_data):
            data['Disk'] = data['Disk'] - background_data['Disk']
            return data

        def correct_electrolyte_res(ORR_data, EIS_data):
            def find_electrolyte_res(data):
                electrolyte_res = data.iloc[(data['Imag']).abs().argsort()[:2]]
                return electrolyte_res['Real'].iloc[0]

            electrolyte_res = find_electrolyte_res(EIS_data)
            ORR_data['Pot'] = ORR_data['Pot'] - (
                    (ORR_data['Disk'] / 1000) * electrolyte_res * float(Config.glob('ELECTRODE_AREA')))
            return ORR_data

        def correct_electrode(data, kalibrierung=0, pH=1):
            data['Pot'] = data['Pot'] - kalibrierung
            return data

        orr_data = data.orr
        orr_data['corrected'] = orr_data['formatted'].copy(deep=True)

        if corr_dict['electrode_corr'] is True:
            correct_electrode(orr_data['corrected'])
            if corr_dict['background_corr'] is True:
                orr_bckg_data = data.orr_bckg
                orr_bckg_data['corrected'] = orr_bckg_data['formatted'].copy(deep=True)
                correct_electrode(orr_bckg_data['corrected'])
                data.set_orr_bckg(orr_bckg_data)

        if corr_dict['background_corr'] is True:
            orr_bckg_data = data.orr_bckg
            orr_bckg_data['corrected'] = orr_bckg_data['formatted'].copy(deep=True)
            correct_background(orr_data['corrected'], orr_bckg_data['corrected'])

        if corr_dict['eis_corr'] is True:
            eis_data = data.eis['formatted']
            correct_electrolyte_res(orr_data['corrected'], eis_data)

        data.set_orr(orr_data)

        half_scan_end = int(len(data.orr['corrected'].index) / 2)
        anodic_scan = data.orr['corrected'].iloc[60:half_scan_end - 60]
        cathodic_scan = data.orr['corrected'].iloc[half_scan_end + 60:-60]
        cathodic_scan = cathodic_scan[::-1]

        data.set_anodic(anodic_scan)
        data.set_cathodic(cathodic_scan)
        plotter.plot_corr_orr()
        return

    # get half-wave potential,
    # current at certain potentials,
    # onset point,
    # Koutecky-Levich analysis and get j lim and j kin
    # Tafel plot analysis and get parameters
    def analyse_data(self, onset_threshold=5, jlim_threshold=0.0002):
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
            Diff_max = input_data['Diff1'].max()
            inflection_point = input_data.query('Diff1 == @Diff_max', inplace=False)
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

            j = find_fixed_potential(input_data, 0.75)
            j_lim = find_jlim(input_data)
            return (j * j_lim) / (j_lim - j)

        def find_Tafel(input):
            j_lim = -abs(input['Disk']).max()
            dat = input.query('0.4 < Pot', inplace=False)
            dat['Tafel'] = (dat['Disk'] * j_lim) / (j_lim - dat['Disk']) * (-1)
            dat['Tafel'] = np.log10(dat['Tafel'])
            dat = dat.iloc[::20, :]
            dat['Diff'] = (dat['Tafel'].diff() / dat['Pot'].diff())
            return dat

        def find_n(input_data, N):
            df = input_data.query('0.1 < Pot < 0.7', inplace=False).copy(deep=True)
            df['n'] = (4 * df['Disk']) / (df['Disk'] + (df['Ring'] / N))
            return df['n'].mean()

        # 'tafel': find_Tafel(scan)
        for scan in [reduced_anodic, reduced_cathodic]:
            parameters.append({'halfwave_pot': find_inflectionpoint(scan),
                               'onset': find_onset(scan),
                               'peroxide_yield': find_peroxide_yield(scan, 0.37),
                               'n': find_n(scan, 0.37),
                               'activity': find_activity(scan)})
        data.set_parameters(parameters)
        self.show_parameters()
        return plotter.plot_parameters(reduced_anodic, reduced_cathodic)

    def show_parameters(self):
        root = self.ids['parameters']
        root.pos_hint = {"top": 0.6, "center_x": 0.82}
        root.children[1].clear_widgets()
        root.children[1].add_widget(Label(text='anodic', bold=True, color=(0, 0, 0, 1)))
        root.children[2].clear_widgets()
        root.children[2].add_widget(Label(text='cathodic', bold=True, color=(0, 0, 0, 1)))
        params = data.parameters
        for i in range(1, 3):
            for key in params[i]:
                val = params[i][key].round(3)
                lbl = Label(text=str(val), color=(0, 0, 0, 1))
                root.children[i].add_widget(lbl)

    # Export corrected data and summary to txt and csv
    # Export all plots to png
    def export_data(self):

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
            write_to_file(data.parameters[1], dir / parameters_ano_file_name)
            write_to_file(data.parameters[2], dir / parameters_cat_file_name)
        except TypeError:
            return


class ScreenTwo(Screen):
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
data = Data()


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


if __name__ == '__main__':
    InterfaceApp().run()
