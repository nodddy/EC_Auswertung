from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput

import pandas as pd
import tkfilebrowser
import matplotlib.pyplot as plt
from pathlib import Path

import DataHandler
import CustomWidgets as CW


class Plotter:
    plot_dict = {
        'orr': {
            'x_label': 'Pot vs. RHE [V]',
            'y_label': 'Disk Current [mA]'
        },

        'eis': {
            'x_label': "Real Z' (\u03A9)",
            'y_label': "Imaginary -Z'' (\u03A9)"
        },

        'orr_analysis': {
            'x_label': 'Pot vs. RHE [V]',
            'y_label': 'Disk Current Density [mA/cm2]'
        },

        'cv': {
            'y_label': 'Current Density [mA/cm2]',
            'x_label': 'Potential vs. RHE [V]'
        },
        'lsv': {
            'y_label': 'Current Density [mA/cm2]',
            'x_label': 'Potential vs. RHE [V]'
        }
    }

    def __init__(self):
        pass

    @staticmethod
    def overwrite_plot(layout_instance, figure=None):
        if figure is None:
            return
        layout_instance.ids['plotter'].clear_widgets()
        canvas = FigureCanvasKivyAgg(figure)
        layout_instance.ids['plotter'].add_widget(canvas)
        return canvas

    def plot(self, layout_instance, plot_name, label, x_data, y_data):
        if not isinstance(x_data, list):
            x_data = [x_data]
            y_data = [y_data]
            label = [label]
        fig, ax = plt.subplots(1, 1)
        for x, y, lbl in zip(x_data, y_data, label):
            ax.plot(x, y, label=lbl)
        ax.set_xlabel(self.plot_dict[plot_name]['x_label'])
        ax.set_ylabel(self.plot_dict[plot_name]['y_label'])
        ax.set_title(plot_name, pad=25, fontdict={'fontsize': 18})
        ax.legend()
        fig_widget = self.overwrite_plot(layout_instance, fig)
        return fig_widget, ax

    def plot_parameters(self, fig_widget, ax_widget, ano_analysis, cat_analysis):

        def get_point(value, dataset, col):
            return dataset.iloc[(dataset[col] - value).abs().argsort()[:1]]

        row_list = []
        for scan in [ano_analysis, cat_analysis]:
            for param in ['halfwave_pot', 'onset_pot', 'cur_lim']:
                col_ident = [id.capitalize() for id in param.split('_') if 'pot' in id or 'cur' in id][0]
                row = get_point(getattr(scan, param), getattr(scan, 'orr'), col_ident)
                row_list.append(row)

        for item in row_list:
            ax_widget.scatter(item['Pot'], item['Cur'])
        return fig_widget.draw()

    def plot_ecsa(self, fig_widget, ax_widget, ecsa_curve):
        if type(ecsa_curve) is not list: ecsa_curve = [ecsa_curve]
        for item in ecsa_curve:
            ax_widget.fill(item['Pot'], item['Cur'])
        return fig_widget.draw()


class ScreenZero(Screen):
    """
    initial navigation screen
    """

    def change_screen(self, screen_name):
        manager.current = str(screen_name)


class ScreenOne(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.clearcolor = (0.9, 0.9, 0.9, 1)
        Window.borderless = False
        self.tab_manager = None

    def on_enter(self, *args):
        """
        sets the first instance in the dict and adds it to the parent FloatLayout. Only does it once at the first time.
        """
        if self.tab_manager is None:
            instance_dict = {
                'screen_orr': 'OrrTabContent',
                'screen_cv': 'CvTabContent',
                'screen_testbench': 'TestbenchTabContent'
            }
            self.tab_manager = CW.TabManager(
                content_cls=eval(instance_dict[self.name]),
                pos_hint={'right': 1, 'top': 1},
                size_hint=(0.92, 1)
            )
            self.children[0].add_widget(self.tab_manager)

    @staticmethod
    def import_data(instance, data_var: str, data_class: str, skip_row: int = 0, delimiter='\t'):
        """ imports ORR, saves as instance variable and plots it """
        path = Path(
            tkfilebrowser.askopenfilename(
                filetypes=[("Textfile", ".txt"), ('Textfile', ".csv")],
                initialdir='C:/Users/Marius/Documents/GitHub/EC_Auswertung/Daten')
        )
        if str(path) == '.' or str(path) == '':
            return False
        try:
            setattr(
                instance,
                data_var,
                eval(f'DataHandler.{data_class}')(
                    path=path,
                    raw_data=pd.read_csv(
                        path,
                        sep=delimiter,
                        skiprows=skip_row
                    )
                ))
        except pd.core.computation.ops.UndefinedVariableError:
            return False
        return True


class OrrTabContent(CW.TabContent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_plot = None
        self.message_list = []
        self.current_orr = None
        self.current_orr_bckg = None
        self.current_eis = None
        self.current_ano_analysis = None
        self.current_cat_analysis = None
        self.current_export = None
        self.parameter_dict = {'halfwave_pot': 'Halfwave Potential [V]',
                               'onset_pot': 'Onset Potential [V]',
                               'cur_lim': 'Limited Current [A/cm^2]',
                               'activity': 'Activity [A/cm^2]',
                               'peroxide_yield': 'Peroxide Yield [%]',
                               'e_transfer': 'Electron Transfer Number'}

    def import_orr(self):
        """ imports ORR, saves as instance variable and plots it """
        if ScreenOne.import_data(self, 'current_orr', 'Orr') is False:
            return
        self.current_plot = plotter.plot(
            self,
            'orr',
            label='Raw ORR',
            x_data=self.current_orr.formatted['Pot'],
            y_data=self.current_orr.formatted['Cur']
        )[0]
        active_screen = manager.get_screen(manager.current)
        active_screen.tab_manager.rename_current_instance(
            new_name=self.current_orr.path.name.split('.')[0]
        )
        return

    def import_orr_bckg(self):
        """ imports ORR and saves as background instance variable """
        if ScreenOne.import_data(self, 'current_orr_bckg', 'OrrBckg') is False:
            return

    def import_eis(self):
        """ imports EIS data and saves it as instance variable """
        if ScreenOne.import_data(self, 'current_eis', 'Eis') is False:
            return

    def correct_orr(self):
        """ corrects the current ORR analysis and plots the corrected ORR curve """
        if self.current_orr is None:
            return
        self.current_orr.correct(orr_bckg=self.current_orr_bckg,
                                 eis=self.current_eis)
        self.current_plot = plotter.plot(self, 'orr', label='Corrected ORR',
                                         x_data=self.current_orr.corrected['Pot'],
                                         y_data=self.current_orr.corrected['Cur'])[0]

    def analyse_orr(self):
        """
        -creates instances of ORR analysis for anodic and cathodic scans and sets them as instance variable
        -clears the current plot and plots anodic and cathodic scans with parameters
        """
        if self.current_orr is None:
            return
        self.current_ano_analysis = DataHandler.OrrAnalysis(orr=self.current_orr.anodic)
        self.current_ano_analysis.stage = 'anodic'
        self.current_cat_analysis = DataHandler.OrrAnalysis(orr=self.current_orr.cathodic)
        self.current_cat_analysis.stage = 'cathodic'
        self.current_plot, axes_widget = plotter.plot(
            self,
            'orr',
            label=['Anodic', 'Cathodic'],
            x_data=[self.current_orr.anodic['Pot'], self.current_orr.cathodic['Pot']],
            y_data=[self.current_orr.anodic['Cur'], self.current_orr.cathodic['Cur']]
        )
        plotter.plot_parameters(
            self.current_plot,
            axes_widget,
            self.current_ano_analysis,
            self.current_cat_analysis
        )
        self.add_parameter_data()

    def export_data(self):
        """ asks for export directory and creates export instances for anodic and cathodic scans """
        if self.current_orr is None:
            return
        export_dir = tkfilebrowser.askopendirname(initialdir='C:/Users/Marius/Documents/GitHub/EC_Auswertung/Daten')
        ano_export = DataHandler.ExportOrr(path=export_dir,
                                           analysis_instance=self.current_ano_analysis)
        ano_export.export_data()
        cat_export = DataHandler.ExportOrr(path=export_dir,
                                           analysis_instance=self.current_cat_analysis)
        cat_export.export_data()

    def add_current_data(self):
        label_list = [f'{data.name}:  {data.path.name}' for data in
                      [self.current_orr, self.current_orr_bckg, self.current_eis] if data is not None]
        self.add_labels(self.ids['data_labels'], label_list)
        return

    def add_parameter_data(self):
        for box_id, instance in {'cathodic': self.current_cat_analysis, 'anodic': self.current_ano_analysis}.items():
            label_list = [str(round(abs(val), 3)) for val in instance.__dict__.values() if isinstance(val, float)]
            self.add_labels(self.ids[box_id], label_list)
        for box_id, instance in {'parameter': self.current_ano_analysis}.items():
            label_list = [self.parameter_dict[key] for key, val in instance.__dict__.items() if isinstance(val, float)]
            self.add_labels(self.ids[box_id], label_list)

    def add_labels(self, label_box, label_list):
        label_box.clear_widgets()
        for text in label_list:
            label_box.add_widget(Label(
                text=text,
                color=(0, 0, 0, 1),
                font_size=14))

    def clear_plot(self):
        self.ids['plotter'].clear_widgets()
        self.current_plot = None

    def safe_plot_to_png(self):
        if self.current_plot is None:
            return
        else:
            self.current_plot.print_png(
                f'{self.current_orr.path.parents[0] / self.current_orr.path.name.split(".")[0]}.png')


class CvTabContent(CW.TabContent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_plot = None
        self.current_cv = None

    def import_cv(self):
        if ScreenOne.import_data(self, 'current_cv', 'Cv') is False:
            return


class TestbenchTabContent(CW.TabContent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_plot = None
        self.message_list = []
        self.current_eis_dict = None
        self.current_cv = None
        self.current_lsv = None
        self.current_cv_analysis = None
        self.current_lsv_analysis = None
        self.current_export = None
        self.parameter_dict = {
            'ecsa': 'EASA [m^2/g_Pt]',
            'h_cross': 'H2 Cross-over [mA/cm^2]'
        }

    def import_lsv(self):
        """ imports LSV and saves as background instance variable """
        if ScreenOne.import_data(self, 'current_lsv', 'Lsv', 3, ',') is False:
            return
        self.current_plot = plotter.plot(
            self,
            'lsv',
            label=[
                'Raw LSV',
                'Corrected LSV'
            ],
            x_data=[
                self.current_lsv.formatted['Pot'],
                self.current_lsv.corrected['Pot']
            ],
            y_data=[
                self.current_lsv.formatted['Cur'],
                self.current_lsv.corrected['Cur']
            ],
        )[0]
        return

    def import_cv(self):
        """ imports CV and saves it as instance variable """
        if ScreenOne.import_data(self, 'current_cv', 'CvTestbench', 3, ',') is False:
            return
        self.current_plot = plotter.plot(
            self,
            'cv',
            label='Raw CV',
            x_data=self.current_cv.formatted['Pot'],
            y_data=self.current_cv.formatted['Cur']
        )[0]
        return

    def correct_cv(self):
        """ corrects the current CV and plots the corrected curve """
        if self.current_cv is None or self.current_lsv is None:
            return
        self.current_cv.correct(
            self.current_lsv.res_slope,
            self.current_lsv.res_intercept
        )
        self.current_plot = plotter.plot(
            self,
            'cv',
            label=[
                'Corrected CV',
                'Raw CV'
            ],
            x_data=[
                self.current_cv.corrected['Pot'],
                self.current_cv.formatted['Pot']
            ],
            y_data=[
                self.current_cv.corrected['Cur'],
                self.current_cv.formatted['Cur']
            ]
        )[0]

    def analyse_cv(self):
        if self.current_cv is None:
            return
        elif self.current_cv.corrected is None:
            cv_df = self.current_cv.formatted
        else:
            cv_df = self.current_cv.corrected

        self.current_cv_analysis = DataHandler.CvAnalysis(cv_df)
        self.current_plot, axes_widget = plotter.plot(
            self,
            'cv',
            label='CV',
            x_data=cv_df['Pot'],
            y_data=cv_df['Cur']
        )
        plotter.plot_ecsa(
            self.current_plot,
            axes_widget,
            self.current_cv_analysis.ecsa_curve
        )

    def analyse_lsv(self):
        if self.current_lsv is None:
            return
        elif self.current_lsv.corrected is None:
            lsv_df = self.current_lsv.formatted
        else:
            lsv_df = self.current_lsv.corrected

        self.current_lsv_analysis = DataHandler.LsvAnalysis(lsv_df)
        self.current_plot, axes_widget = plotter.plot(
            self,
            'lsv',
            label='LSV',
            x_data=lsv_df['Pot'],
            y_data=lsv_df['Cur']
        )

    def export_data(self):
        """ asks for export directory and creates export instances for anodic and cathodic scans """
        if self.current_cv is None and self.current_lsv is None:
            return
        export_dir = tkfilebrowser.askopendirname()
        data_instances = [inst for inst in [self.current_lsv, self.current_cv] if inst is not None]
        analysis_instances = [inst for inst in [self.current_lsv_analysis, self.current_cv_analysis] if
                              inst is not None]
        export = DataHandler.ExportTestbench(
            path=export_dir,
            analysis_instances=analysis_instances,
            data_instances=data_instances
        )
        export.export_data()

    def add_current_data(self):
        label_list = [f'{data.name}:  {data.path.name}' for data in
                      [self.current_cv, self.current_lsv] if data is not None]
        self.add_labels(self.ids['data_labels'], label_list)
        return

    def add_parameter_data(self):
        label_list = []
        for instance in [self.current_cv_analysis, self.current_lsv_analysis]:
            if instance is not None:
                label_list.extend(
                    [str(round(abs(val), 3)) for val in instance.__dict__.values() if isinstance(val, float)])
        self.add_labels(self.ids['para_value'], label_list)
        label_list = []
        for instance in [self.current_cv_analysis, self.current_lsv_analysis]:
            if instance is not None:
                label_list.extend(
                    [self.parameter_dict[key] for key, val in instance.__dict__.items() if isinstance(val, float)])
        self.add_labels(self.ids['parameter'], label_list)

    def add_labels(self, label_box, label_list):
        label_box.clear_widgets()
        for text in label_list:
            label_box.add_widget(Label(
                text=text,
                color=(0, 0, 0, 1),
                font_size=14))

    def clear_plot(self):
        self.ids['plotter'].clear_widgets()
        self.current_plot = None

    def safe_plot_to_png(self):
        if self.current_plot is None:
            return
        else:
            self.current_plot.print_png(
                f'{self.current_orr.path.parents[0] / self.current_orr.path.name.split(".")[0]}.png')


class SettingsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_edit = None

    def on_enter(self, *args):
        """
        calls the update from config file on each enter on the screen
        """
        self.update_settings_from_config()

    def update_settings_from_config(self):
        """
        reads the config file and adds new widgets for each header and their individual configs with values.
        """
        self.ids['box_settings'].clear_widgets()
        height_hint = 1 / (1 + max([len(config._sections[key]) for key in config._sections]))
        for key in config._sections:
            box = BoxLayout(orientation='vertical',
                            size_hint_y=(1 + len(config._sections[key])) * height_hint,
                            pos_hint={'top': 1})
            box.add_widget(Label(text=str(key),
                                 color=(0, 0, 0, 1),
                                 font_size=14,
                                 bold=True))
            for setting, val in config._sections[key].items():
                inner_box = BoxLayout(orientation='horizontal')
                inner_box.add_widget(Label(text=str(setting),
                                           color=(0, 0, 0, 1),
                                           font_size=14))
                inner_box.add_widget(Label(text=str(val),
                                           color=(0, 0, 0, 1),
                                           font_size=14,
                                           size_hint=(0.5, 0.6),
                                           pos_hint={'center_y': 0.5}))
                inner_box.add_widget(CW.ImageButton(img='img/edit_button.png',
                                                    on_press=self.edit_setting,
                                                    size_hint=(0.6, 0.6),
                                                    pos_hint={'center_x': 0.5, 'center_y': 0.5}))
                box.add_widget(inner_box)
            self.ids['box_settings'].add_widget(box)

    def edit_setting(self, btn):
        """
        checks if a edit is currently done, if not then clears all labels and buttons from the current edited row
        and adds a new label and textinput with the new save button
        """
        if self.current_edit is not None:
            return
        box = btn.parent
        self.current_edit = box
        setting = box.children[2].text
        val = box.children[1].text
        box.clear_widgets()
        box.add_widget(Label(text=str(setting),
                             color=(0, 0, 0, 1),
                             font_size=14))
        box.add_widget(TextInput(text=str(val),
                                 size_hint=(0.5, 0.6),
                                 pos_hint={'center_y': 0.5},
                                 multiline=False,
                                 on_text_validate=self.save_setting))
        box.add_widget(CW.ImageButton(img='img/save_button.png',
                                      on_press=self.save_setting,
                                      size_hint=(0.6, 0.6),
                                      pos_hint={'center_x': 0.5, 'center_y': 0.5}))
        return

    def save_setting(self, btn):
        """
        takes the button instance from the save button and writes to the config file according to the new value.
        Afterwards resets the current edit and renews all setting widgets from the new config file
         """
        box = btn.parent
        key = [widget for widget in box.parent.children if isinstance(widget, Label)][0].text
        config._sections[key][box.children[2].text] = box.children[1].text
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        self.current_edit = None
        self.update_settings_from_config()
        return


class Navigation(Popup):
    def __init__(self, screen_manager, **kwargs):
        super(Navigation, self).__init__(**kwargs)
        self.manager = screen_manager
        return


class Manager(ScreenManager):
    screen_one = ObjectProperty(None)
    screen_two = ObjectProperty(None)


class UIApp(App):
    Window.maximize()

    def build(self):
        global manager
        global plotter
        global config
        config = DataHandler.Config.initialize()
        plotter = Plotter()
        manager = Manager()
        return manager

    @staticmethod
    def open_navigation():
        navigation = Navigation(manager)
        return navigation.open()


if __name__ == '__main__':
    UIApp().run()
