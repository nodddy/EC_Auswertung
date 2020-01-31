import pandas as pd
from configparser import ConfigParser
import os


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

    def __init__(self, raw_data, path):
        self.raw = Data.rename_header(raw_data)
        self.path = path
        self.formatted = None
        self.corrected = None

    @staticmethod
    def rename_header(raw_data):
        header_dict = {
            'WE(1).Current (A)': 'Cur',
            'WE(2).Current (A)': 'Ring',
            'WE(1).Potential(V)': 'Pot',
            "Z' (\u03A9)": 'Real',
            "Z'' (\u03A9)": 'Imag'}
        return raw_data.rename(columns=header_dict, inplace=True)


class Orr(Data):

    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)
        self.cathodic = None
        self.anodic = None
        self.format()

    def format(self):
        raw_orr = self.raw.copy(deep=True)
        scan1 = raw_orr.query('Scan == 2').reset_index(inplace=True)
        scan2 = raw_orr.query('Scan == 3').reset_index(inplace=True)
        formatted_orr = pd.DataFrame()
        formatted_orr['Pot'] = ((scan1['Pot'] + scan2['Pot']) / 2)
        formatted_orr['Disk'] = ((scan1['Cur'] + scan2['Cur']) / 2)
        try:
            formatted_orr['Ring'] = ((scan1['Ring'] + scan2['Ring']) / 2)
        except KeyError:
            pass

        half_scan_end = int(len(formatted_orr.index) / 2)
        self.anodic = formatted_orr.iloc[40:half_scan_end - 40]
        self.cathodic = formatted_orr.iloc[half_scan_end + 40:-40][::-1]
        self.formatted = formatted_orr.reset_index(inplace=True)
        return

    def correct(self, orr_bckg=None, eis_intercept=None):
        orr = self.formatted.copy(deep=True)
        if eis_intercept is not None:
            orr['Pot'] = orr['Pot'] - (orr['Cur'] * eis_intercept)

            if orr_bckg is not None:
                orr_bckg['Pot'] = orr_bckg['Pot'] - (orr_bckg['Cur'] * eis_intercept)

        if orr_bckg is not None:
            orr['Cur'] = orr['Cur'] - orr_bckg['Cur']

        self.corrected = orr
        return


class OrrBckg(Data):

    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)
        self.format()

    def format(self):
        raw_orr = self.raw.copy(deep=True)
        scan1 = raw_orr.query('Scan == 2').reset_index(inplace=True)
        scan2 = raw_orr.query('Scan == 3').reset_index(inplace=True)
        formatted_orr = pd.DataFrame()
        formatted_orr['Pot'] = ((scan1['Pot'] + scan2['Pot']) / 2)
        formatted_orr['Disk'] = ((scan1['Cur'] + scan2['Cur']) / 2)
        self.formatted = formatted_orr.reset_index(inplace=True)
        return


class Eis(Data):

    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)
        self.x_intercept = self.find_intercept()

    def find_intercept(self):
        df = self.formatted
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
        self.orr['Diff1'] = (self.orr['Cur'].diff() / self.orr['Pot'].diff()).dropna()
        return

    def find_halfwave_pot(self):
        diff_max = self.orr['Diff1'].max()
        inflection_point = self.orr.query(f'Diff1 == {diff_max}', inplace=False)
        return inflection_point['Pot'].iloc[0]

    def find_onset_pot(self):
        cur_max = abs(self.orr['Cur']).max()
        cur_min = abs(self.orr['Cur']).min()
        normalised_orr = pd.DataFrame()
        normalised_orr['Cur'] = (abs(self.orr['Cur']) - cur_min) / (cur_max - cur_min)
        normalised_orr['Pot'] = self.orr['Pot']
        return normalised_orr.query('Cur < 0.02', inplace=False)['Pot'].iloc[0]

    def find_peroxide_yield(self):
        N = float(Config.glob('RRDE_N'))
        orr_df = self.orr.query('0.1 < Pot < 0.75', inplace=False)
        orr_df['PeroxideYield'] = ((2 * (abs(self.orr['Ring'] / N))) / (
                abs(self.orr['Cur']) + (abs(self.orr['Ring']) / N))) * 100
        return orr_df['PeroxideYield'].mean()

    def find_e_transfer(self):
        N = float(Config.glob('RRDE_N'))
        orr_df = self.orr.query('0.1 < Pot < 0.75', inplace=False)
        orr_df['n'] = (4 * orr_df['Cur'].abs()) / (orr_df['Cur'].abs() + (orr_df['Ring'] / N))
        return orr_df['n'].mean()

    def find_cur_lim(self):
        return -abs(self.orr['Cur']).max()

    def find_activity(self):
        def find_fixed_potential(input_df, potential):
            orr_df = input_df.copy(deep=True)
            return orr_df.iloc[(orr_df['Pot'] - potential).abs().argsort()[:2]]['Cur'].mean()

        j = find_fixed_potential(self.orr, float(Config.glob('J_KIN_POTENTIAL')))
        return (j * self.cur_lim) / (self.cur_lim - j)


class CvAnalysis(Analysis):
    def __init__(self, **kwargs):
        super(Analysis, self).__init__(**kwargs)


class Export:
    def __init__(self):
        pass
