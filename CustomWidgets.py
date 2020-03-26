from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button


class TabManager(RelativeLayout):
    def __init__(self, content_cls, **kwargs):
        super().__init__(**kwargs)
        self.content_cls = content_cls
        self.instances = {}
        self.tab_box = self.build_tab_box()
        self.current_instance = self.add_instance(hide=False)[0]

    def build_tab_box(self):
        tab_bar_box = BoxLayout(
            pos_hint={'top': 1, 'center_x': 0.5},
            size_hint=(0.8, 0.08)
        )
        tab_bar_box.add_widget(ImageButton(
            img='img/add_button.png',
            id='add_btn',
            size_hint=(0.6, 0.6),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            on_press=self.add_instance
        ))
        self.add_widget(tab_bar_box)
        return tab_bar_box

    def build_instance_buttons(self):
        """
        clears the tab widget and construcuts all buttons again with the currently active tab having a special border.
        """
        self.tab_box.clear_widgets()
        for key in self.instances.keys():
            try:
                if key == self.current_instance:  # decides which border the button gets (blue for active)
                    button_img = 'img/active_tab_button.png'
                else:
                    button_img = 'img/passive_tab_button.png'
            except AttributeError:
                button_img = 'img/active_tab_button.png'
            self.tab_box.add_widget(Button(  # adds each button
                text=str(key),
                id=str(key),
                size_hint=(0.2, 0.7),
                pos_hint={'center_x': 0.5, 'center_y': 0.5},
                bold=True,
                font_size=18,
                background_normal=button_img,
                on_press=self.open_instance)  # sets the button instance as first argument
            )
        self.tab_box.add_widget(ImageButton(
            img='img/add_button.png',
            on_press=self.add_instance,
            size_hint=(None, 0.6),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        ))

    def open_instance(self, instance):
        """
        takes the instance of the pressed tab button and toggles the associated OrrTabContent widget into view and  sets
        the new current instance variable. If the same button as the active instance is pressed, nothing happens.
        It constructs the tab widget anew to reflect the change as current tab with the border.
        """
        if instance.id == self.current_instance:
            return

        self.instances[self.current_instance].toggle_instance_widget()
        self.instances[self.current_instance].leave_instance()
        self.current_instance = instance.id
        self.build_instance_buttons()
        self.instances[instance.id].toggle_instance_widget()
        self.instances[instance.id].on_tab_activation()

    def rename_current_instance(self, new_name):
        """
        renames the current instance tab to name given as an argument and constructs the tab widget anew.
        """
        if new_name in self.instances.keys():
            return
        self.instances[new_name] = self.instances.pop(self.current_instance)
        self.current_instance = new_name
        self.build_instance_buttons()

    def add_instance(self, *args, hide=True):
        """
        adds a new tab button to the parent FloatLayout and registers it in the instances dict.
        """
        new_key = f'Tab {1 + len([key for key in self.instances.keys() if "Tab" in key])}'
        if hide is True:
            right_pos_hint = -1
        else:
            right_pos_hint = 1
        new_instance = self.content_cls(
            pos_hint={'right': right_pos_hint, 'top': 0.92},
            size_hint=(1, 1)
        )
        self.instances[new_key] = new_instance
        self.add_widget(new_instance)
        self.build_instance_buttons()
        return new_key, new_instance


class TabContent(RelativeLayout):
    def on_tab_activation(self):
        if self.current_plot is not None:
            self.ids['plotter'].clear_widgets()
            self.ids['plotter'].add_widget(self.current_plot)
        self.open_instance()

    def toggle_instance_widget(self):
        """
        toggles the current OrrScreen widget in and out of view.
        """
        if self.pos_hint == {'right': 1, 'top': 0.92}:
            self.pos_hint = {'right': -1, 'top': 0.92}
        else:
            self.pos_hint = {'right': 1, 'top': 0.92}

    def open_instance(self):
        pass

    def leave_instance(self):
        pass


class ImageButton(ButtonBehavior, Image):
    """
    new button class thats takes an image path as arg and creates a button with the image
    """

    def __init__(self, img='', **kwargs):
        super().__init__(**kwargs)
        self.source = img
