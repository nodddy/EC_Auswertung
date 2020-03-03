import pandas as pd
from configparser import ConfigParser
import os
from pathlib import Path


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
    def header(cls, key):
        return cls.configParser.get('DATA_HEADER', key)

    @classmethod
    def exp_params(cls, key):
        return cls.configParser.get('EXPERIMENTAL_PARAMETERS', key)

    @classmethod
    def units(cls, key):
        return cls.configParser.get('UNITS', key)


class Data:

    def __init__(self, raw_data, path):
        self.raw = Data.convert_units(Data.rename_header(raw_data))
        self.path = path
        self.formatted = None
        self.corrected = None

    @staticmethod
    def rename_header(raw_data):
        header_dict = {
            str(Config.header('CURRENT_HEADER')): 'Cur',
            str(Config.header('RING_CURRENT_HEADER')): 'Ring',
            str(Config.header('POTENTIAL_HEADER')): 'Pot',
            "Z' (\u03A9)": 'Real',
            "-Z'' (\u03A9)": 'Imag'}
        raw_data.rename(columns=header_dict, inplace=True)
        return raw_data

    @staticmethod
    def convert_units(raw_data):
        unit_dict = {'Cur': (lambda x: float(x) * float(Config.exp_params('ELECTRODE_AREA'))),
                     'Ring': (lambda x: float(x) * float(Config.exp_params('ELECTRODE_AREA')))}
        for key, func in unit_dict.items():
            if str(key) in list(raw_data):
                raw_data[key] = raw_data[key].map(func)
        return raw_data


class Orr(Data):

    def __init__(self, raw_data, path):
        super().__init__(raw_data, path)
        self.cathodic = None
        self.anodic = None
        self.format()
        self.name = 'ORR'

    def format(self):
        raw_orr = self.raw.copy(deep=True)
        scan1 = raw_orr.query('Scan == 2')
        scan1.reset_index(inplace=True)
        scan2 = raw_orr.query('Scan == 3')
        scan2.reset_index(inplace=True)
        formatted_orr = pd.DataFrame()
        formatted_orr['Pot'] = ((scan1['Pot'] + scan2['Pot']) / 2)
        formatted_orr['Cur'] = ((scan1['Cur'] + scan2['Cur']) / 2)
        try:
            formatted_orr['Ring'] = ((scan1['Ring'] + scan2['Ring']) / 2)
        except KeyError:
            pass

        half_scan_end = int(len(formatted_orr.index) / 2)
        self.anodic = formatted_orr.iloc[40:half_scan_end - 40]
        self.cathodic = formatted_orr.iloc[half_scan_end + 40:-40][::-1]
        formatted_orr.reset_index(inplace=True)
        self.formatted = formatted_orr
        return

    def correct(self, orr_bckg=None, eis=None):
        orr = self.formatted.copy(deep=True)
        if eis is not None:
            orr['Pot'] = orr['Pot'] - (orr['Cur'] * float(Config.exp_params('ELECTRODE_AREA')) * eis.x_intercept)

            if orr_bckg is not None:
                orr_bckg.formatted['Pot'] = orr_bckg.formatted['Pot'] - (
                        orr_bckg.formatted['Cur'] * float(Config.exp_params('ELECTRODE_AREA')) * eis.x_intercept)

        if orr_bckg is not None:
            orr['Cur'] = orr['Cur'] - orr_bckg.formatted['Cur']

        self.corrected = orr
        half_scan_end = int(len(orr.index) / 2)
        self.anodic = orr.iloc[40:half_scan_end - 40]
        self.cathodic = orr.iloc[half_scan_end + 40:-40][::-1]
        return


class OrrBckg(Data):

    def __init__(self, raw_data, path):
        super().__init__(raw_data, path)
        self.format()
        self.name = 'ORR background'

    def format(self):
        raw_orr = self.raw.copy(deep=True)
        scan1 = raw_orr.query('Scan == 2')
        scan1.reset_index(inplace=True)
        scan2 = raw_orr.query('Scan == 3')
        scan2.reset_index(inplace=True)
        formatted_orr = pd.DataFrame()
        formatted_orr['Pot'] = ((scan1['Pot'] + scan2['Pot']) / 2)
        formatted_orr['Cur'] = ((scan1['Cur'] + scan2['Cur']) / 2)
        formatted_orr.reset_index(inplace=True)
        self.formatted = formatted_orr
        return


class Eis(Data):

    def __init__(self, raw_data, path):
        super().__init__(raw_data, path)
        self.x_intercept = self.find_intercept()
        self.name = 'EIS'

    def find_intercept(self):
        df = self.raw.copy(deep=True)
        return df.iloc[(df['Imag']).abs().argsort()[:2]]['Real'].iloc[0]


class Cv(Data):

    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)

    def format(self):
        raw_cv = self.raw.copy(deep=True)
        scan2 = raw_cv.iloc[int(len(raw_cv) / 3):int((len(raw_cv) / 3) * 2)].reset_index(inplace=True)
        scan3 = raw_cv.iloc[int((len(raw_cv) / 3) * 2):int(len(raw_cv))].reset_index(inplace=True)
        formatted_cv = pd.DataFrame()
        formatted_cv['Cur'] = (scan2['Cur'] + scan3['Cur']) / 2
        formatted_cv['Pot'] = (scan2['Pot'] + scan3['Pot']) / 2
        self.formatted = formatted_cv
        return


class Analysis:

    def __init__(self):
        pass


class OrrAnalysis(Analysis):

    def __init__(self, orr, **kwargs):
        """
        creates analysis instance with the orr data and analyses all parameters
        :param orr: pandas dataframe with orr data, either anodic or cathodic
        """
        super(Analysis, self).__init__(**kwargs)
        self.orr = orr.copy(deep=True)
        self.differentiate()
        self.halfwave_pot = self.find_halfwave_pot()
        self.onset_pot = self.find_onset_pot()
        self.cur_lim = self.find_cur_lim()
        self.activity = self.find_activity()
        self.tafel_slope = None
        self.tafel_data = None
        self.peroxide_yield = self.find_peroxide_yield()
        self.e_transfer = self.find_e_transfer()

    def differentiate(self):
        """
        differentiates the orr data
        :return: Nothing
        """
        self.orr['Diff1'] = (self.orr['Cur'].diff() / self.orr['Pot'].diff()).dropna()
        return

    def find_halfwave_pot(self):
        """
        finds the halfwave potential by finding the maximum of the first differential ergo the inflectionpoint
        :return: one potential value
        """
        diff_max = self.orr['Diff1'].max()
        inflection_point = self.orr.query(f'Diff1 == {diff_max}', inplace=False)
        return float(inflection_point['Pot'].iloc[0])

    def find_onset_pot(self):
        """
        finds the onset potential by normalising orr data and finding the first point from high to low potential,
        which is higher than 2% current
        :return: one potential value
        """
        cur_max = abs(self.orr['Cur']).max()
        cur_min = abs(self.orr['Cur']).min()
        normalised_orr = pd.DataFrame()
        normalised_orr['Cur'] = (abs(self.orr['Cur']) - cur_min) / (cur_max - cur_min)
        normalised_orr['Pot'] = self.orr['Pot']
        return float(normalised_orr.query('Cur < 0.02', inplace=False)['Pot'].iloc[0])

    def find_peroxide_yield(self):
        """
        finds the peroxide yield by the formula with absolute current values:
        X=( 2*i_ring / N ) / ( i_disk + (i_ring / N) ) * 100
        :return: the mean value of all values over potential area
        """
        N = float(Config.exp_params('RRDE_N'))
        orr_df = self.orr.query('0.1 < Pot < 0.75', inplace=False)
        orr_df['PeroxideYield'] = ((2 * (abs(self.orr['Ring'] / N))) / (
                abs(self.orr['Cur']) + (abs(self.orr['Ring']) / N))) * 100
        return float(orr_df['PeroxideYield'].mean())

    def find_e_transfer(self):
        N = float(Config.exp_params('RRDE_N'))
        orr_df = self.orr.query('0.1 < Pot < 0.75', inplace=False)
        orr_df['n'] = (4 * orr_df['Cur'].abs()) / (orr_df['Cur'].abs() + (orr_df['Ring'] / N))
        return float(orr_df['n'].mean())

    def find_cur_lim(self):
        return float(-abs(self.orr['Cur']).max())

    def find_activity(self):
        def find_fixed_potential(input_df, potential):
            orr_df = input_df.copy(deep=True)
            return orr_df.iloc[(orr_df['Pot'] - potential).abs().argsort()[:2]]['Cur'].mean()

        j = find_fixed_potential(self.orr, float(Config.exp_params('J_KIN_POTENTIAL')))
        return float((j * self.cur_lim) / (self.cur_lim - j))


class CvAnalysis(Analysis):
    def __init__(self, **kwargs):
        super(Analysis, self).__init__(**kwargs)


class Export:
    def __init__(self, path):
        self.path = Path(path)

    def rename_header(self, df):
        header_dict = {
            'Cur': 'Current [A]',
            'Ring': 'Ring Current [A]',
            'Pot': 'Potential vs. RHE [V]',
            "Z' (\u03A9)": 'Real',
            "-Z'' (\u03A9)": 'Imag'}
        df.rename(columns=header_dict, inplace=True)
        return df

    def change_units(self, df):
        return

    def str_to_file(self, export_str, file_name):
        """
        exports a string to file
        :param export_str: the file to be exported as string
        :return: the file
        """
        with open(self.path / file_name, 'w') as file:
            file.write(export_str)
        return file


class ExportOrr(Export):
    def __init__(self, path, analysis_instance):
        super().__init__(path)
        self.orr = self.rename_header(analysis_instance.orr)
        self.instance = analysis_instance

    def parameters_to_export_str(self):
        """
        writes a pandas dataframe as .csv to a directory provided
        :param analysis_instance: the analysis instance containing all parameters as instance variables
        :return: string to be exported
        """
        export_str = 'Parameter\tValue'
        for key, val in self.instance.__dict__.items():
            if isinstance(val, float):
                export_str = f'{export_str}\n{key}\t{val}'
        return export_str

    def export_data(self):
        self.orr.to_csv(self.path / f'{self.instance.stage}_orr.txt',
                        columns=['Potential vs. RHE [V]', 'Current [A]'],
                        index=False)
        self.str_to_file(export_str=self.parameters_to_export_str(),
                         file_name=f'{self.instance.stage}_parameters.txt')
