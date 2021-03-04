import pandas as pd
from configparser import ConfigParser
import os
from pathlib import Path
import numpy as np
from scipy import integrate
from sklearn.linear_model import LinearRegression
from unidecode import unidecode


class Config:
    """Interact with configuration variables."""

    configParser = ConfigParser()
    configFilePath = (os.path.join(os.getcwd(), 'config.ini'))

    @classmethod
    def initialize(cls):
        """Start config by reading config.ini."""
        cls.configParser.read(cls.configFilePath)
        return cls.configParser

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
    def exp_params_bool(cls, key):
        return cls.configParser.getboolean('EXPERIMENTAL_PARAMETERS', key)

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
            "-Z'' (\u03A9)": 'Imag',
            str(Config.header('TESTBENCH_POTENTIAL_HEADER')): 'Pot',
            str(Config.header('TESTBENCH_CURRENT_HEADER')): 'Cur',
            str(Config.header('TESTBENCH_TIME_HEADER')): 'Time',
            str(Config.header('TESTBENCH_EIS_REAL_HEADER')): 'Real',
            str(Config.header('POROSITY_APPL.PRESSURE_HEADER')): 'ApplPressure',
            str(Config.header('POROSITY_INTR.VOLUME_HEADER')): 'IntrVolume'
        }
        raw_data.rename(columns=lambda x: unidecode(x), inplace=True)
        raw_data.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
        header_dict = {k.replace(' ', ''): v for (k, v) in header_dict.items()}
        return raw_data.rename(columns=header_dict)

    @staticmethod
    def convert_units(raw_data):
        unit_dict = {'Cur': (lambda x: (float(x) * 1000) / float(Config.exp_params('ELECTRODE_AREA'))),
                     'Ring': (lambda x: (float(x) * 1000) / float(Config.exp_params('ELECTRODE_AREA')))}
        for key, func in unit_dict.items():
            if str(key) in list(raw_data):
                raw_data[key] = raw_data[key].map(func)
        return raw_data


class Orr(Data):

    def __init__(self, raw_data, path):
        super().__init__(raw_data, path)
        self.cathodic = None
        self.anodic = None
        self.formatted = self.format()
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
        return formatted_orr.reset_index()

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
        self.formatted = self.format()
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
        return formatted_orr.reset_index()


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
        super().__init__(**kwargs)
        self.formatted = self.format()

    def format(self):
        raw_cv = self.raw.copy(deep=True)
        scan2 = raw_cv.iloc[int(len(raw_cv) / 3):int((len(raw_cv) / 3) * 2)].reset_index()
        scan3 = raw_cv.iloc[int((len(raw_cv) / 3) * 2):int(len(raw_cv))].reset_index()
        formatted_cv = pd.DataFrame()
        formatted_cv['Cur'] = (scan2['Cur'] + scan3['Cur']) / 2
        formatted_cv['Pot'] = (scan2['Pot'] + scan3['Pot']) / 2
        return formatted_cv.reset_index()


class Lsv(Data):

    def __init__(self, raw_data, path):
        super().__init__(raw_data, path)
        self.formatted = self.format()
        self.res_slope, self.res_intercept = self.find_resistance()
        self.name = 'LSV'
        self.corrected = self.correct()

    def format(self):
        return self.raw.copy(deep=True)

    def find_resistance(self):
        pot_range = (0.3, 0.4)
        df_slice = self.formatted.query(
            f'{pot_range[0]} < Pot < {pot_range[1]}',
            inplace=False
        )
        x = np.asarray(df_slice['Pot']).astype(np.float64).reshape(-1, 1)
        y = np.asarray(df_slice['Cur']).astype(np.float64)
        model = LinearRegression().fit(x, y)
        return model.coef_[0], model.intercept_

    def correct(self):
        corrected_lsv = self.formatted.copy(deep=True)
        corrected_lsv['Cur'] = corrected_lsv['Cur'] - (
                corrected_lsv['Pot']
                * self.res_slope
                + self.res_intercept
        )
        return corrected_lsv


class CvTestbench(Data):

    def __init__(self, raw_data, path):
        super().__init__(raw_data, path)
        if Config.exp_params_bool('CV_FORMAT_AUTO') is True:
            self.formatted = self.format_auto()
        else:
            self.formatted = self.format_manual(Config.exp_params('CV_FORMAT_TIME_LIMIT'))
        self.corrected = None
        self.name = 'CV'

    def format_auto(self):
        """

        """
        raw_cv = self.raw.copy(deep=True)
        max_pot = raw_cv['Pot'].max()
        min_pot = raw_cv['Pot'].min()
        max_pot_df = raw_cv.query(
            f'Pot == {max_pot}',
            inplace=False
        )
        min_pot_df = raw_cv.query(
            f'Pot == {min_pot}',
            inplace=False
        )
        half_scan_duration = np.subtract(max_pot_df['Time'].to_numpy().reshape(-1, 1),
                                         min_pot_df['Time'].to_numpy().reshape(1, -1)).min() + 0.1
        last_time_val = raw_cv['Time'].iloc[raw_cv['Time'].last_valid_index()]
        scan_number = round(last_time_val / (half_scan_duration * 2), 0) - float(Config.exp_params('CV_SCAN_USED'))
        lower_time_limit = (last_time_val - ((half_scan_duration * 2) * (scan_number + 1)))
        upper_time_limit = last_time_val - ((half_scan_duration * 2) * scan_number)
        scan = raw_cv.query(
            f'{lower_time_limit} < Time < {upper_time_limit}',
            inplace=False
        )
        return scan.reset_index(inplace=False)

    def format_manual(self, time_limit):
        raw_cv = self.raw.copy(deep=True)
        return raw_cv.query(
            f'{time_limit[0]} < Time < {time_limit[1]}',
            inplace=False
        )

    def correct(self, slope, intercept):
        corrected_cv = self.formatted.copy(deep=True)
        corrected_cv['Cur'] = corrected_cv['Cur'] - (
                corrected_cv['Pot']
                * slope
                + intercept
        )
        self.corrected = corrected_cv


class Porosity(Data):
    def __init__(self, raw_data, path):
        super().__init__(raw_data, path)
        self.name = 'Porosity'
        self.parameter_dict = self.parameters_from_file()

    def parameters_from_file(self):
        """
        opens file from path and iterated through to search for parameter results and Experimental Data.
        Parameters are packed into dict as strings and data is packed into pandas dataframe, which get returned.
        """
        with open(self.path, 'r') as f:
            file_lines = f.read().splitlines()
        first_index = None
        last_index = None
        param_dict = {}
        for index, line in enumerate(file_lines):
            if 'RESULTS WITHOUT COMPRESSIB. CORR.' in line:
                first_index = index
            if first_index is not None and '' == line:
                last_index = index
                break
        for item in [item.replace(' ', '').split(';') for item in file_lines[first_index + 1:last_index]]:
            i = iter(item)
            param_dict.update(dict(zip(i, i)))
        return param_dict


class Analysis:
    def __init__(self):
        pass


class OrrAnalysis(Analysis):

    def __init__(self, orr, **kwargs):
        """
        creates analysis instance with the orr data and analyses all parameters
        :param orr: pandas dataframe with orr data, either anodic or cathodic
        """
        super().__init__(**kwargs)
        self.orr = orr.copy(deep=True)
        self.orr['Diff1'] = self.differentiate()
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
        :return: first differential
        """
        return (self.orr['Cur'].diff() / self.orr['Pot'].diff()).dropna()

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
    def __init__(self, cv_df, **kwargs):
        super().__init__(**kwargs)
        self.pot_range = (0.3, 0.6)
        self.ecsa, self.ecsa_curve = self.find_ecsa(
            cv_df.query('Cur > 0',
                        inplace=False)
        )

    def find_ecsa(self, cv_df):
        def integrate_curve(curve_y, curve_x):
            return integrate.trapz(curve_y, curve_x)

        def find_curve_section(cv_data, lower_pot, upper_pot):
            min_point = cv_data.query(f'{lower_pot} < Pot < {upper_pot}', inplace=False)
            min_point_cur = min_point['Cur'].min()
            min_point_pot = min_point.query(f'Cur == {min_point_cur}')['Pot'].iloc[0]
            return cv_data.query(
                f' Pot <= {min_point_pot} and Cur >= {min_point_cur}',
                inplace=False
            )

        ecsa_curve = find_curve_section(
            cv_df,
            self.pot_range[0],
            self.pot_range[1]
        )
        ecsa_area = integrate_curve(
            ecsa_curve['Cur'],
            ecsa_curve['Pot']
        )
        capa_area = integrate_curve(
            np.full(len(ecsa_curve),
                    ecsa_curve['Cur'].min()),
            ecsa_curve['Pot']
        )
        if cv_df['Cur'].iloc[0] < 0:
            ecsa_curve['Cur'] = ecsa_curve['Cur'] * -1
        scan_rate = float(Config.exp_params('CV_SCAN_RATE'))
        ecsa = (ecsa_area - capa_area) / (2.1 * scan_rate * float(Config.exp_params('PT_LOADING')))
        return float(ecsa), ecsa_curve


class LsvAnalysis(Analysis):
    def __init__(self, lsv_df, **kwargs):
        super().__init__(**kwargs)
        self.pot_range = (0.3, 0.4)
        self.h_cross = self.find_h_cross(lsv_df)

    def find_h_cross(self, lsv_df):
        slice = lsv_df.query(f'{self.pot_range[0]} < Pot < {self.pot_range[1]}')
        return float(slice['Cur'].mean())


class Export:
    def __init__(self, path):
        self.path = Path(path)

    def rename_header(self, df):
        header_dict = {
            'Cur': 'Current Density [mA/cm^2]',
            'Ring': 'Ring Current Density [mA/cm^2]',
            'Pot': 'Potential vs. RHE [V]',
            "Z' (\u03A9)": 'Real',
            "-Z'' (\u03A9)": 'Imag'}
        return df.rename(columns=header_dict)

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
        self.instance = analysis_instance
        self.orr = self.rename_header(analysis_instance.orr.copy(deep=True))

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
        """
        writes ORR dataframes to csv and parameters as txt file to the path variable of the instance
        """

        self.orr.to_csv(self.path / f'{self.instance.stage}_orr.txt',
                        columns=['Potential vs. RHE [V]', 'Current [A]'],
                        index=False)
        self.str_to_file(export_str=self.parameters_to_export_str(),
                         file_name=f'{self.instance.stage}_parameters.txt')


class ExportTestbench(Export):
    def __init__(self, path, analysis_instances, data_instances):
        super().__init__(path)
        if type(data_instances) is not list: data_instances = [data_instances]
        self.data_instances = data_instances
        if type(analysis_instances) is not list: analysis_instances = [analysis_instances]
        self.analysis_instances = analysis_instances

    def parameters_to_export_str(self):
        """
        writes a pandas dataframe as .csv to a directory provided
        :param analysis_instance: the analysis instance containing all parameters as instance variables
        :return: string to be exported
        """
        export_str = 'Parameter\tValue'
        for instance in self.analysis_instances:
            for key, val in instance.__dict__.items():
                if isinstance(val, float):
                    export_str = f'{export_str}\n{key}\t{val}'
        return export_str

    def export_data(self):
        """
        writes ORR dataframes to csv and parameters as txt file to the path variable of the instance
        """
        for instance in self.data_instances:
            export_df = instance.formatted.copy(deep=True)
            export_df = self.rename_header(export_df)
            export_df.to_csv(
                self.path / f'{instance.path.name.split(".")[0]}_raw.txt',
                columns=['Potential vs. RHE [V]', 'Current Density [mA/cm^2]'],
                index=False
            )
            if instance.corrected is not None:
                export_df = instance.corrected.copy(deep=True)
                export_df = self.rename_header(export_df)
                export_df.to_csv(
                    self.path / f'{instance.path.name.split(".")[0]}_corrected.txt',
                    columns=['Potential vs. RHE [V]', 'Current Density [mA/cm^2]'],
                    index=False
                )

        self.str_to_file(
            export_str=self.parameters_to_export_str(),
            file_name=f'{self.data_instances[0].path.name.split("_")[1]}_parameters.txt'
        )
