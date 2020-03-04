from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior

import pandas as pd
import tkfilebrowser
import matplotlib.pyplot as plt
from pathlib import Path

import DataHandler


class Plotter:
    plot_dict = {'orr': {'x_label': 'Pot vs. RHE [V]',
                         'y_label': 'Disk Current [mA]'},

                 'eis': {'x_label': "Real Z' (\u03A9)",
                         'y_label': "Imaginary -Z'' (\u03A9)"},

                 'orr_analysis': {'x_label': 'Pot vs. RHE [V]',
                                  'y_label': 'Disk Current Density [mA/cm2]'},

                 'cv': {'y_label': 'Current Density [mA/cm2]',
                        'x_label': 'Potential vs. RHE [V]'}}

    def __init__(self):
        pass

    @staticmethod
    def overwrite_plot():
        active_screen = manager.get_screen(manager.current)
        active_screen.ids['plotter'].clear_widgets()
        active_screen.ids['plotter'].add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def plot(self, plot_name, label, x_data, y_data):
        plt.xlabel(self.plot_dict[plot_name]['x_label'])
        plt.ylabel(self.plot_dict[plot_name]['y_label'])
        plt.title(plot_name, pad=25, fontdict={'fontsize': 18})
        plt.plot(x_data, y_data * 1000, label=label)
        plt.legend()
        return self.overwrite_plot()

    def plot_parameters(self, ano_analysis, cat_analysis):
        def get_point(value, dataset, col):
            return dataset.iloc[(dataset[col] - value).abs().argsort()[:1]]

        row = get_point(ano_analysis.halfwave_pot, ano_analysis.orr, 'Pot')
        plt.scatter(row['Pot'], row['Cur'] * 1000, label='anodic halfwave potential')
        row = get_point(ano_analysis.onset_pot, ano_analysis.orr, 'Pot')
        plt.scatter(row['Pot'], row['Cur'] * 1000, label='anodic onset potential')
        row = get_point(cat_analysis.halfwave_pot, cat_analysis.orr, 'Pot')
        plt.scatter(row['Pot'], row['Cur'] * 1000, label='cathodic halfwave potential')
        row = get_point(cat_analysis.onset_pot, cat_analysis.orr, 'Pot')
        plt.scatter(row['Pot'], row['Cur'] * 1000, label='cathodic onset potential')
        row = get_point(ano_analysis.cur_lim, ano_analysis.orr, 'Cur')
        plt.scatter(row['Pot'], row['Cur'] * 1000, label='anodic limited current')
        row = get_point(cat_analysis.cur_lim, cat_analysis.orr, 'Cur')
        plt.scatter(row['Pot'], row['Cur'] * 1000, label='cathodic limited current')
        return


class ScreenOne(Screen):

    def __init__(self, **kwargs):
        super(ScreenOne, self).__init__(**kwargs)
        Window.clearcolor = (0.9, 0.9, 0.9, 1)
        Window.borderless = False
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
        orr_path = Path(
            tkfilebrowser.askopenfilename(filetypes=[("Textfile", "*.txt")],
                                          initialdir='C:/Users/Marius/Documents/GitHub/EC_Auswertung/Daten'))
        if str(orr_path) == '.':
            return
        try:
            self.current_orr = DataHandler.Orr(path=orr_path,
                                               raw_data=pd.read_csv(orr_path, sep='\t'))
        except pd.core.computation.ops.UndefinedVariableError:
            return
        plotter.plot('orr', label='Raw ORR',
                     x_data=self.current_orr.formatted['Pot'],
                     y_data=self.current_orr.formatted['Cur'])
        return

    def import_orr_bckg(self):
        """ imports ORR and saves as background instance variable """
        orr_bckg_path = Path(
            tkfilebrowser.askopenfilename(filetypes=[("Textfile", "*.txt")],
                                          initialdir='C:/Users/Marius/Documents/GitHub/EC_Auswertung/Daten'))
        if str(orr_bckg_path) == '.':
            return
        self.current_orr_bckg = DataHandler.OrrBckg(path=orr_bckg_path,
                                                    raw_data=pd.read_csv(orr_bckg_path, sep='\t'))

    def import_eis(self):
        """ imports EIS data and saves it as instance variable """
        eis_path = Path(
            tkfilebrowser.askopenfilename(filetypes=[("Textfile", "*.txt")],
                                          initialdir='C:/Users/Marius/Documents/GitHub/EC_Auswertung/Daten'))
        if str(eis_path) == '.':
            return
        self.current_eis = DataHandler.Eis(path=eis_path,
                                           raw_data=pd.read_csv(eis_path, sep='\t'))

    def correct_orr(self):
        """ corrects the current ORR analysis and plots the corrected ORR curve """
        self.current_orr.correct(orr_bckg=self.current_orr_bckg,
                                 eis=self.current_eis)
        plotter.plot('orr', label='Corrected ORR',
                     x_data=self.current_orr.corrected['Pot'],
                     y_data=self.current_orr.corrected['Cur'])

    def analyse_orr(self):
        """
        -creates instances of ORR analysis for anodic and cathodic scans and sets them as instance variable
        -clears the current plot and plots anodic and cathodic scans with parameters
        """
        self.current_ano_analysis = DataHandler.OrrAnalysis(orr=self.current_orr.anodic)
        self.current_ano_analysis.stage = 'anodic'
        self.current_cat_analysis = DataHandler.OrrAnalysis(orr=self.current_orr.cathodic)
        self.current_cat_analysis.stage = 'cathodic'
        plt.cla()
        plotter.plot('orr', label='Anodic',
                     x_data=self.current_orr.anodic['Pot'],
                     y_data=self.current_orr.anodic['Cur'])
        plotter.plot('orr', label='Cathodic',
                     x_data=self.current_orr.cathodic['Pot'],
                     y_data=self.current_orr.cathodic['Cur'])
        plotter.plot_parameters(self.current_ano_analysis, self.current_cat_analysis)
        self.add_parameter_data()

    def export_data(self):
        """ asks for export directory and creates export instances for anodic and cathodic scans """
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


class ScreenTwo(Screen):
    pass


class ScreenThree(Screen):
    pass


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
                inner_box.add_widget(ImageButton(img='img/edit_button.png',
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
        box.add_widget(ImageButton(img='img/save_button.png',
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


class ImageButton(ButtonBehavior, Image):
    """
    new button class thats takes an image path as arg and creates a button with the image
    """
    def __init__(self, img, **kwargs):
        super().__init__(**kwargs)
        self.source = img


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
