import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,  # Creates the GUI application
    QMainWindow,  # Creates the main window
    QVBoxLayout,  # Creates vertical box
    QSlider,  # Creates slider
    QLabel,  # Creates label
    QWidget,  # Creates widget
    QHBoxLayout,  # Creates horizontal box
    QPushButton,  # Creates button
    QLineEdit,  # Creates text input
    QFileDialog,  # Creates file dialog
    QMessageBox,  # Creates message box
    QAction,  # Creates action for menu items
    QFrame,  # Creates frame
    QDialog,  # Creates dialog
    QDialogButtonBox,  # Creates button box for dialog
    QComboBox,  # Creates dropdown
    QScrollArea,  # Creates scroll area
    QCheckBox,  # Creates check box
)
from PyQt5.QtCore import Qt, QCoreApplication, QProcess, QTimer
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)  # Allows embedding Matplotlib plots in PyQt
from matplotlib.figure import Figure  # Used to create the Matplotlib figure
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from xspec import *
import os

# Define a list of colors for plotting spectra
SPECTRUM_COLORS = ['black', 'red', 'lime', 'blue', 'cyan', 'magenta', 'yellow', 'orange']

# Load the relxill models
AllModels.lmod('relxill')

# Global constants
WINDOW_TITLE = "XSPEC Model"
WINDOW_GEOMETRY = (100, 100, 1000, 600)
PLOT_X_AXIS = 'keV'
PLOT_X_LOG = True
PLOT_Y_LOG = True
PLOT_Y_MIN_FACTOR = 10**(-15)
PLOT_Y_MAX_FACTOR = 1.15
SLIDER_PRECISION_FACTOR = 1000
DEFAULT_MODEL_NAME = "e.g., powerlaw"
DEFAULT_PLOT_TYPE = 'model'
DEFAULT_DATA_PLOT_OPTION = 'data'
DEFAULT_COLOR_MAP = "plasma"
DEFAULT_SPACING = "linear"
DEFAULT_CURVE_COUNT = 10
Plot.area = True
Plot.setRebin(10, 10, -1)

class MainWindow(QMainWindow):
    """
    MainWindow class provides a GUI for interacting with XSPEC models.
    It allows users to input a model name, adjust parameters using sliders,
    and visualize the model plot.
    """

    def __init__(self):
        """
        Initialize the main window, set up the layout, and connect signals.
        """
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(*WINDOW_GEOMETRY)  # Set window dimensions
        self.model_name = None

        # Create a menubar
        menubar = self.menuBar()

        # Add 'File' menu
        file_menu = menubar.addMenu('File')
        load_model_xcm_action = QAction('Load Model as XCM', self)  # Renamed action
        save_plot_action = QAction('Save Plot', self)
        save_xcm_action = QAction('Save Parameters as XCM', self)  # New action
        load_data_xcm_action = QAction('Load Data as XCM', self)  # New action
        load_plot_style_action = QAction('Load Plot Style File', self)  # New action
        exit_action = QAction('Exit', self)
        restart_action = QAction('Restart', self)  # New action
        file_menu.addAction(load_model_xcm_action)  # Updated action
        file_menu.addAction(save_plot_action)
        file_menu.addAction(save_xcm_action)  # Add new action
        file_menu.addAction(load_data_xcm_action)  # Add new action
        file_menu.addAction(load_plot_style_action)  # Add new action
        file_menu.addAction(restart_action)  # Add new action
        file_menu.addAction(exit_action)

        # Add 'View' menu
        view_menu = menubar.addMenu('View')
        freeze_axes_action = QAction('Freeze Axes', self, checkable=True)
        use_textboxes_action = QAction('Use Textboxes for Parameters', self, checkable=True)
        include_background_action = QAction('Include Background For Data', self, checkable=True)
        set_axes_limits_action = QAction('Set Axes Limits', self)
        plot_components_action = QAction('Plot Different Components', self, checkable=True)
        select_plot_action = QAction('Select What to Plot', self)
        plot_n_times_action = QAction('Plot Same Curve N Times', self)  # New action
        show_frozen_action = QAction('Show Frozen Parameters', self, checkable=True)  # New action
        plot_data_action = QAction('Plot Data', self)  # New action
        view_menu.addAction(freeze_axes_action)
        view_menu.addAction(use_textboxes_action)
        view_menu.addAction(include_background_action)
        view_menu.addAction(set_axes_limits_action)
        view_menu.addAction(plot_components_action)
        view_menu.addAction(select_plot_action)
        view_menu.addAction(plot_n_times_action)  # Add new action
        view_menu.addAction(show_frozen_action)  # Add new action
        view_menu.addAction(plot_data_action)  # Add new action
        set_axes_limits_action.triggered.connect(self.set_axes_limits)
        plot_n_times_action.triggered.connect(self.plot_same_curve_n_times)  # Connect new action
        show_frozen_action.triggered.connect(lambda: self.toggle_option('Show Frozen Parameters'))  # Connect new action
        plot_data_action.triggered.connect(self.open_plot_data_dialog)  # Connect new action

        # Connect actions to methods (placeholders)
        load_model_xcm_action.triggered.connect(self.load_model_as_xcm)  # Updated connection
        save_plot_action.triggered.connect(self.save_plot)
        save_xcm_action.triggered.connect(self.save_parameters_as_xcm)  # Connect new action
        load_data_xcm_action.triggered.connect(self.load_data_as_xcm)  # Connect new action
        load_plot_style_action.triggered.connect(self.load_plot_style)  # Connect new action
        exit_action.triggered.connect(self.close)
        restart_action.triggered.connect(self.restart_application)  # Connect new action

        freeze_axes_action.triggered.connect(lambda: self.toggle_option('Freeze Axes'))
        use_textboxes_action.triggered.connect(lambda: self.toggle_option('Use Textboxes for Parameters'))
        include_background_action.triggered.connect(lambda: self.toggle_option('Include Background'))
        plot_components_action.triggered.connect(lambda: self.toggle_option('Plot Different Components'))
        select_plot_action.triggered.connect(self.open_select_plot_dialog)

        # Initialize layout components
        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()

        # Create a scroll area for the left panel
        left_panel_scroll_area = QScrollArea()
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel)
        left_panel_scroll_area.setWidget(left_panel_widget)
        left_panel_scroll_area.setWidgetResizable(True)

        # Model selection components
        self.model_label = QLabel("Enter Model:")
        self.model_textbox = QLineEdit()
        self.model_textbox.setPlaceholderText(DEFAULT_MODEL_NAME)
        self.model_textbox.returnPressed.connect(self.load_model)
        left_panel.addWidget(self.model_label)
        left_panel.addWidget(self.model_textbox)

        # Button to generate plot
        self.plot_button = QPushButton("Generate Plot")
        self.plot_button.clicked.connect(self.generate_plot)
        left_panel.addWidget(self.plot_button)

        # Checkbox to rescale plot
        self.rescale_checkbox = QCheckBox("Rescale Plot")
        self.rescale_checkbox.stateChanged.connect(self.rescale_plot)
        self.rescale_checkbox.hide()  # Initially hide the checkbox
        left_panel.addWidget(self.rescale_checkbox)

        # Initialize lists for sliders and labels
        self.param_sliders = []
        self.param_labels = []
        self.model_params = []
        self.scale_factors = []
        self.precision_factors = []
        self.param_textboxes = []

        # Add stretch to left panel
        left_panel.addStretch()

        # Add the left panel scroll area to the main layout
        main_layout.addWidget(left_panel_scroll_area, stretch=1)

        # Plot area setup
        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.add_subplot(111)

        # Add components to main layout
        main_layout.addWidget(self.canvas, stretch=3)

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Initialize background visibility state
        self.include_background = False
        self.is_model_loaded = False
        self.what_to_plot = DEFAULT_PLOT_TYPE
        self.use_textboxes_selected = False
        self.plot_components_selected = False
        self.freeze_axes_selected = False
        self.is_data_loaded = False
        self.show_frozen_parameters = False
        self.is_data_loaded = False

    def load_model(self):
        """
        Load the XSPEC model entered in the textbox, initialize UI elements (sliders or textboxes)
        for model parameters, and update the plot.
        """
        if not self.is_model_loaded:
            # Retrieve the model name from the textbox and initialize the model
            self.model_name = self.model_textbox.text().strip()
            self.models = [Model(self.model_name)]

        try:
            # Clear existing sliders or textboxes
            self.clear_sliders()

            # Clear existing horizontal lines and labels
            left_panel = self.centralWidget().layout().itemAt(0).widget().widget().layout()
            for i in reversed(range(left_panel.count())):
                item = left_panel.itemAt(i)
                widget = item.widget()
                if (isinstance(widget, QFrame) and widget.frameShape() == QFrame.HLine) or (isinstance(widget, QLabel) and widget.text().startswith('Component:')):
                    left_panel.removeWidget(widget)
                    widget.deleteLater()

            # Initialize lists to store labels and indices for model parameters
            self.labels = []
            self.indices = []
            self.labels_in_comps = []
            self.comps = []

            for model in self.models:
                labels = []
                indices = []
                labels_in_comps = []
                i = 1

                # Extract component names from the model
                comps = model.componentNames
                for component in comps:
                    # Get parameter names for each component
                    labels_in_component = getattr(model, component).parameterNames
                    labels_in_comps.append(labels_in_component)
                    for par in labels_in_component:
                        # Add parameter to list if it's not linked
                        if getattr(getattr(model, component), par).link == '':
                            indices.append(i)
                            if par == 'norm':
                                labels.append(par + '_' + component[0])
                            else:
                                labels.append(par)
                            i += 1

                # Store extracted labels and indices
                self.labels.append(labels)
                self.indices.append(indices)
                self.labels_in_comps.append(labels_in_comps)
                self.comps.append(comps)

                # Initialize model parameters list
                self.model_params = []
                for i in range(len(labels_in_comps)):
                    for j in range(len(labels_in_comps[i])):
                        param = getattr(getattr(model, comps[i]), labels_in_comps[i][j])
                        if getattr(self, 'show_frozen_parameters', True) or (not param.frozen and param.link == ''):
                            self.model_params.append(param)

                counter = 0
                for i in range(len(labels_in_comps)):
                    component_label = QLabel(f"Component: {comps[i]}")
                    left_panel.insertWidget(counter * 2 + 1, component_label)
                    counter += 1

                    for j in range(len(labels_in_comps[i])):
                        param = getattr(getattr(model, comps[i]), labels_in_comps[i][j])
                        label = QLabel(f"{param.name}: {param.values[0]:.3f}")  # Display parameter value

                        if param in self.model_params:
                            if hasattr(self, 'use_textboxes_selected') and self.use_textboxes_selected:
                                # Use textboxes for parameter input
                                if not self.param_textboxes:
                                    for slider in self.param_sliders:
                                        left_panel = self.centralWidget().layout().itemAt(0).widget().widget().layout()
                                        left_panel.removeWidget(slider)
                                        slider.deleteLater()
                                    self.param_sliders.clear()
                                    for label in self.param_labels:
                                        left_panel = self.centralWidget().layout().itemAt(0).widget().widget().layout()
                                        left_panel.removeWidget(label)
                                        label.deleteLater()
                                    self.param_labels.clear()
                                    
                                label = QLabel(f"{param.name}: {param.values[0]:.3f}")
                                textbox = QLineEdit(str(param.values[0]))
                                textbox.editingFinished.connect(lambda tb=textbox, p=param: setattr(p, 'values', [float(tb.text())]))
                                left_panel = self.centralWidget().layout().itemAt(0).widget().widget().layout()
                                left_panel.insertWidget(counter * 2, label)
                                left_panel.insertWidget(counter * 2 + 1, textbox)
                                self.param_textboxes.append(textbox)
                                self.param_labels.append(label)
                            else:
                                # Use sliders for parameter input
                                for textbox in self.param_textboxes:
                                    left_panel = self.centralWidget().layout().itemAt(0).widget().widget().layout()
                                    left_panel.removeWidget(textbox)
                                    textbox.deleteLater()
                                self.param_textboxes.clear()
                                slider, scale_factor, precision_factor = self.create_slider(param)
                                slider.valueChanged.connect(lambda value, p=param, l=label, sf=scale_factor, pf=precision_factor: self.update_param_label(value, p, l, sf, pf))
                                left_panel = self.centralWidget().layout().itemAt(0).widget().widget().layout()
                                left_panel.insertWidget(counter * 2, label)
                                left_panel.insertWidget(counter * 2 + 1, slider)
                                self.param_sliders.append(slider)
                                self.scale_factors.append(scale_factor)
                                self.precision_factors.append(precision_factor)
                                self.param_labels.append(label)

                            counter += 1
                    # Insert a horizontal line after each component's parameters
                    line = QFrame()
                    line.setFrameShape(QFrame.HLine)
                    line.setFrameShadow(QFrame.Sunken)
                    left_panel = self.centralWidget().layout().itemAt(0).widget().widget().layout()
                    left_panel.insertWidget(counter * 2, line)

            # Hide the label and textbox after loading the model
            self.model_label.hide()
            self.model_textbox.hide()
            # Show the rescale checkbox
            self.rescale_checkbox.show()
            # Update the plot with the new model
            self.update_plot()

        except Exception as e:
            # Display an error message if model loading fails
            self.model_label.setText(f"Enter Model: (Error: {str(e)})")

    def create_lambda(self, param, textbox):
        """
        Create a lambda function to update parameter values from a textbox.

        Parameters:
        param: The model parameter to update.
        textbox: The QLineEdit widget containing the new parameter value.

        Returns:
        A lambda function that updates the parameter value.
        """
        return lambda: setattr(param, 'values', [float(textbox.text())])

    def clear_sliders(self):
        """
        Clear all sliders and labels from the layout, and reset parameter lists.
        """
        left_panel = self.centralWidget().layout().itemAt(0).widget().widget().layout()
        for slider, label in zip(self.param_sliders, self.param_labels):
            left_panel.removeWidget(slider)
            left_panel.removeWidget(label)
            slider.deleteLater()
            label.deleteLater()

        for textbox, label in zip(self.param_textboxes, self.param_labels):
            left_panel.removeWidget(textbox)
            left_panel.removeWidget(label)
            textbox.deleteLater()
            label.deleteLater()

        self.param_sliders.clear()
        self.param_labels.clear()
        self.model_params.clear()
        self.scale_factors.clear()
        self.precision_factors.clear()
        self.param_textboxes.clear()

    def create_slider(self, param):
        """
        Create a slider for a given model parameter, scaled for precision.

        Parameters:
        param: The model parameter for which the slider is created.

        Returns:
        slider: The created QSlider object.
        scale_factor: The factor used to scale the parameter value.
        precision_factor: The factor used to increase slider precision.
        """
        max_value = param.values[5]
        scale_factor = 10 ** int(np.floor(np.log10(max_value))) if max_value > 0 else 1

        # Increase precision by scaling to a smaller range
        precision_factor = SLIDER_PRECISION_FACTOR  # Adjust this to increase slider precision
        min_value = int(param.values[2] / scale_factor * precision_factor)
        max_value = int(param.values[5] / scale_factor * precision_factor)
        initial_value = int(param.values[0] / scale_factor * precision_factor)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(initial_value)
        slider.setSingleStep(1)
        return slider, scale_factor, precision_factor

    def update_param_label(self, value, param, label, scale_factor, precision_factor):
        """
        Update the label to reflect the current value of the parameter.

        Parameters:
        - value: The current value of the slider.
        - param: The parameter associated with the slider.
        - label: The QLabel to update.
        - scale_factor: The scale factor for adjusting slider values.
        - precision_factor: The precision factor for slider values.
        """
        scaled_value = (value / precision_factor) * scale_factor
        param.values = [scaled_value] + param.values[1:]
        label.setText(f"{param.name}: {scaled_value:.3e}")

    def generate_plot(self):
        if self.is_data_loaded:
            self.plot_data()
        elif self.plot_components_selected:
            self.plot_different_components()
        else:
            # Ensure the model is set for the plot
            if not hasattr(self, 'models'):
                self.load_model()
            self.update_plot()


    def update_plot(self):
        """
        Generate the plot using the current XSPEC model parameters.
        """
        Plot.device = '/null'  # Suppress plot output
        Plot.xAxis = PLOT_X_AXIS
        Plot.xLog = PLOT_X_LOG
        Plot.yLog = PLOT_Y_LOG

        if AllData.nSpectra > 0:
            Plot("ldata")
        else:
            Plot(self.what_to_plot)  # Load the model

        # Extract data from XSPEC plot
        x = Plot.x()
        y = Plot.model()

        # Store current axes limits
        x_limits = self.ax.get_xlim()
        y_limits = self.ax.get_ylim()

        # Clear previous plot
        self.ax.clear()

        # Generate the plot
        self.ax.plot(x, y, label=self.what_to_plot)
        self.ax.set_xlabel("Energy (keV)")
        if self.what_to_plot == "model":
            self.ax.set_ylabel(r"$\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1}$")
        elif self.what_to_plot == "emodel":
            self.ax.set_ylabel(r"$\rm keV\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
        elif self.what_to_plot == "eemodel":
            self.ax.set_ylabel(r"$\rm keV^2\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
        self.ax.set_title(f"Plot of {self.model_name} {self.what_to_plot}")
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        if AllData.nSpectra >= 1:
            for i in range(AllData.nSpectra):
                self.ax.errorbar(self.xs[i], self.ys[i], xerr=self.xerrs[i], yerr=self.yerrs[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                if self.backs[i] is not None and self.include_background:
                    self.ax.scatter(self.xs[i], self.backs[i], marker='*', label=f'Background {i+1}', color=SPECTRUM_COLORS[i])

        self.ax.legend()

        # Freeze axes if option is selected
        if (hasattr(self, 'freeze_axes_selected') and self.freeze_axes_selected):
            self.ax.set_xlim(x_limits)
            self.ax.set_ylim(y_limits)
        else:
            self.ax.relim()
            self.ax.autoscale_view()

        # Rescale plot if the checkbox is checked
        if self.rescale_checkbox.isChecked():
            self.rescale_plot()

        # Refresh canvas
        self.canvas.draw()

    def save_plot(self):
        """
        Save the current plot image to a file selected by the user.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", "", "PNG Files (*.png);;All Files (*)", options=options)
        if file_path:
            self.canvas.figure.savefig(file_path)
            QMessageBox.information(self, "Save Plot", f"Plot saved to {file_path}")

    def toggle_option(self, option_name):
        """
        Toggle the selection state of a given option.

        Parameters:
        - option_name (str): The name of the option to toggle.
        """
        if option_name == 'Freeze Axes':
            self.freeze_axes_selected = not getattr(self, 'freeze_axes_selected', False)
        elif option_name == 'Use Textboxes for Parameters':
            self.use_textboxes_selected = not getattr(self, 'use_textboxes_selected', False)
        elif option_name == 'Plot Different Components':
            self.plot_components_selected = not getattr(self, 'plot_components_selected', False)
        elif option_name == 'Include Background':
            self.include_background = not getattr(self, 'include_background', False)
        elif option_name == 'Show Frozen Parameters':
            self.show_frozen_parameters = not getattr(self, 'show_frozen_parameters', False)
        self.load_model()

    def save_parameters_as_xcm(self):
        """
        Save the current model parameters to an XCM file selected by the user.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Parameters as XCM", "", "XCM Files (*.xcm);;All Files (*)", options=options)
        if file_path:
            Xset.save(file_path, info='a')
            QMessageBox.information(self, "Save XCM File", f"Parameters saved to {file_path} using XSPEC")

    def load_model_as_xcm(self):
        """
        Open an XCM file and load the model parameters into the application.
        """
        self.is_model_loaded = True
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open XCM File", "", "XCM Files (*.xcm);;All Files (*)", options=options)
        if file_path:
            path_head, path_tail = os.path.split(file_path)
            current_dir = os.getcwd()
            os.chdir(path_head)
            Xset.restore(path_tail)
            os.chdir(current_dir)
            i = 1
            self.models = []
            while True:
                try:
                    model = AllModels(i)
                    self.models.append(model)
                    i += 1
                except Exception:
                    break

            QMessageBox.information(self, "Open XCM File", f"Model parameters loaded from {file_path}")
            self.load_model()  # Update the UI with the loaded parameters

    def load_data_as_xcm(self):
        """
        Load data from an XCM file and plot it on the canvas.
        """
        # Open a file dialog to select an XCM file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open XCM File", "", "XCM Files (*.xcm)")
        if not file_path:
            return  # Exit if no file is selected

        self.is_data_loaded = True
        self.what_to_plot = 'ldata'  # Set plot type to 'ldata'

        path_head, path_tail = os.path.split(file_path)
        current_dir = os.getcwd()
        os.chdir(path_head)
        Xset.restore(path_tail)
        os.chdir(current_dir)

        # Check if models are loaded and update instance attributes
        self.models = []
        i = 1
        while True:
            try:
                model = AllModels(i)
                self.models.append(model)
                i += 1
            except Exception:
                break

        if self.models:
            self.model_name = self.models[0].name  # Get the model name
            print(f"Loaded models: {[model.name for model in self.models]}")  # Print the loaded model names
            self.is_model_loaded = True
            self.load_model()  # Update sliders and labels with the loaded models
        else:
            print("No models loaded from the XCM file.")  # Print warning if no models are loaded

        self.plot_data()

    def set_axes_limits(self):
        """
        Open a dialog to set the axes limits.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle('Set Axes Limits')
        layout = QVBoxLayout()

        # Create textboxes for x and y limits
        x_min_label = QLabel('X Min:')
        x_min_textbox = QLineEdit()
        x_max_label = QLabel('X Max:')
        x_max_textbox = QLineEdit()
        y_min_label = QLabel('Y Min:')
        y_min_textbox = QLineEdit()
        y_max_label = QLabel('Y Max:')
        y_max_textbox = QLineEdit()

        # Add widgets to layout
        layout.addWidget(x_min_label)
        layout.addWidget(x_min_textbox)
        layout.addWidget(x_max_label)
        layout.addWidget(x_max_textbox)
        layout.addWidget(y_min_label)
        layout.addWidget(y_min_textbox)
        layout.addWidget(y_max_label)
        layout.addWidget(y_max_textbox)

        # Add OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.apply_axes_limits(
            x_min_textbox.text() if x_min_textbox.text() else str(self.ax.get_xlim()[0]),
            x_max_textbox.text() if x_max_textbox.text() else str(self.ax.get_xlim()[1]),
            y_min_textbox.text() if y_min_textbox.text() else str(self.ax.get_ylim()[0]),
            y_max_textbox.text() if y_max_textbox.text() else str(self.ax.get_ylim()[1]),
            dialog
        ))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)
        dialog.exec_()

    def apply_axes_limits(self, x_min, x_max, y_min, y_max, dialog):
        """
        Apply the axes limits from the dialog.

        Parameters:
        - x_min (str): The minimum x-axis limit.
        - x_max (str): The maximum x-axis limit.
        - y_min (str): The minimum y-axis limit.
        - y_max (str): The maximum y-axis limit.
        - dialog (QDialog): The dialog instance to close upon successful application.
        """
        try:
            if x_min:
                self.ax.set_xlim(left=float(x_min))
            if x_max:
                self.ax.set_xlim(right=float(x_max))
            if y_min:
                self.ax.set_ylim(bottom=float(y_min))
            if y_max:
                self.ax.set_ylim(top=float(y_max))
            self.canvas.draw()
            dialog.accept()
            self.freeze_axes_selected = True
            AllModels.setEnergies(f'{x_min} {x_max} 1000 log')
            self.update_plot()  # Redraw the plot with the new axes limits
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numbers for the axes limits.')

    def plot_different_components(self):
        """
        Plot the different components of the model separately.
        """
        if not hasattr(self, 'models') or not self.models:
            QMessageBox.warning(self, 'No Model Loaded', 'Please load a model first.')
            return

        # Store current axes limits
        x_limits = self.ax.get_xlim()
        y_limits = self.ax.get_ylim()

        # Clear previous plot
        self.ax.clear()

        for model in self.models:
            # Store original model parameters
            original_params = {}

            for component in model.componentNames:
                # Check if the component has a 'norm' attribute
                if hasattr(getattr(model, component), 'norm'):
                    # Store original normalization value
                    original_params[component] = getattr(model, component).norm.values[0]

            for component in model.componentNames:
                # Check if the component has a 'norm' attribute
                if hasattr(getattr(model, component), 'norm'):
                    # Set all other components' normalization to 0
                    for other_component in model.componentNames:
                        if other_component != component and hasattr(getattr(model, other_component), 'norm'):
                            getattr(model, other_component).norm = 0

                    # Plot the data
                    Plot(self.what_to_plot)
                    x = Plot.x()
                    y = Plot.model()

                    # Plot the component
                    self.ax.plot(x, y, label=f'{component}', color=SPECTRUM_COLORS[model.componentNames.index(component) % len(SPECTRUM_COLORS)])

                    # Restore original normalization values
                    for comp, norm in original_params.items():
                        getattr(model, comp).norm = norm

        # Plot data if available
        if AllData.nSpectra >= 1:
            for i in range(AllData.nSpectra):
                self.ax.errorbar(self.xs[i], self.ys[i], xerr=self.xerrs[i], yerr=self.yerrs[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])

        # Set plot labels and title
        self.ax.set_xlabel('Energy (keV)')
        if self.what_to_plot == "model":
            self.ax.set_ylabel(r"$\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1}$")
        elif self.what_to_plot == "emodel":
            self.ax.set_ylabel(r"$\rm keV\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
        elif self.what_to_plot == "eemodel":
            self.ax.set_ylabel(r"$\rm keV^2\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
        self.ax.set_title('Plot of Different Model Components')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.legend()

        # Restore axes limits if option is selected
        if hasattr(self, 'freeze_axes_selected') and self.freeze_axes_selected:
            self.ax.set_xlim(x_limits)
            self.ax.set_ylim(y_limits)
        else:
            self.ax.relim()
            self.ax.autoscale_view()

        # Refresh canvas
        self.canvas.draw()

    def open_select_plot_dialog(self):
        """
        Open a dialog to select what to plot.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle('Select What to Plot')
        layout = QVBoxLayout()

        # Create a dropdown for plot options
        dropdown = QComboBox()
        dropdown.addItems(['model', 'emodel', 'eemodel'])
        dropdown.setCurrentText(self.what_to_plot)

        # Connect dropdown selection to update the attribute
        dropdown.currentTextChanged.connect(lambda text: setattr(self, 'what_to_plot', text))

        # Add OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: [dialog.accept(), self.update_plot()])
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addWidget(dropdown)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        dialog.exec_()

    def plot_same_curve_n_times(self):
        """
        Display a dialog to input parameters and plot the same curve multiple times.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Plot same curve N times")

        layout = QVBoxLayout()

        # Compute the parameter-to-component dictionary dynamically
        param_to_component = {}
        for i, component in enumerate(self.comps):
            for j, param in enumerate(self.labels_in_comps[i]):
                param_to_component[f'{param}_{component}'] = component

        # Dropdown for selecting parameter
        param_label = QLabel("Select parameter:")
        param_combo = QComboBox()
        param_combo.addItems(list(param_to_component.keys()))  # Assuming self.labels contains model parameters
        layout.addWidget(param_label)
        layout.addWidget(param_combo)

        # Choice for spacing
        spacing_label = QLabel("Spacing:")
        spacing_combo = QComboBox()
        spacing_combo.addItems(["linear", "logarithmic"])
        layout.addWidget(spacing_label)
        layout.addWidget(spacing_combo)

        # Dropdown for selecting color map
        cmap_label = QLabel("Select color map:")
        cmap_combo = QComboBox()
        cmap_combo.addItems(["plasma", "rainbow", "viridis", "inferno", "magma", "cividis"])
        layout.addWidget(cmap_label)
        layout.addWidget(cmap_combo)

        # Input for parameter min
        param_min_label = QLabel("Parameter min:")
        param_min_input = QLineEdit()
        layout.addWidget(param_min_label)
        layout.addWidget(param_min_input)

        # Input for parameter max
        param_max_label = QLabel("Parameter max:")
        param_max_input = QLineEdit()
        layout.addWidget(param_max_label)
        layout.addWidget(param_max_input)

        # Input for number of curves
        n_label = QLabel("Number of curves:")
        n_input = QLineEdit()
        layout.addWidget(n_label)
        layout.addWidget(n_input)

        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        def on_accept():
            selected_param = param_combo.currentText()
            param_min = float(param_min_input.text())
            param_max = float(param_max_input.text())
            n = int(n_input.text())
            spacing = spacing_combo.currentText()
            cmap_name = cmap_combo.currentText()

            if not hasattr(self, 'model'):
                QMessageBox.warning(self, 'No Model Loaded', 'Please load a model first.')
                return

            # Clear previous plot
            self.ax.clear()

            # Determine color normalization based on spacing
            if spacing == "linear":
                norm = mcolors.Normalize(vmin=param_min, vmax=param_max)
            else:
                norm = mcolors.LogNorm(vmin=param_min, vmax=param_max)

            # Generate colors
            values = np.linspace(param_min, param_max, n)
            cmap = cm.get_cmap(cmap_name)
            colors = cmap(norm(values))

            # Plot the same curve N times with specified spacing and colors
            for i in range(n):
                # Adjust the model parameter
                if spacing == "linear":
                    param_value = param_min + i * (param_max - param_min) / (n - 1)
                else:
                    param_value = param_min * (param_max / param_min) ** (i / (n - 1))

                # Use the dictionary to get the component name
                component_name = param_to_component.get(selected_param)
                param_name = selected_param.split('_')[0]
                param_obj = getattr(getattr(self.model, component_name), param_name)
                param_obj.values = [param_value] + param_obj.values[1:]

                # Recalculate the model using self.model
                Plot(self.what_to_plot)
                x = Plot.x()
                y = Plot.model()

                # Plot the curve
                self.ax.plot(x, y, label=f'{selected_param}={param_value:.2f}', linestyle='--', color=colors[i])

            # Restore original parameter values
            param_obj.values[0] = param_min

            # Set plot labels and title
            self.ax.set_xlabel('Energy (keV)')
            if self.what_to_plot == "model":
                self.ax.set_ylabel(r"$\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1}$")
            elif self.what_to_plot == "emodel":
                self.ax.set_ylabel(r"$\rm keV\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
            elif self.what_to_plot == "eemodel":
                self.ax.set_ylabel(r"$\rm keV^2\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
            self.ax.set_title('Repeated Plot of Model')
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
            self.ax.legend()

            # Refresh canvas
            self.canvas.draw()

            dialog.accept()

        button_box.accepted.connect(on_accept)
        button_box.rejected.connect(dialog.reject)

        dialog.exec_()

    def restart_application(self):
        """
        Restart the application to its initial state.
        """
        self.close()
        QCoreApplication.quit()
        QProcess.startDetached(sys.executable, sys.argv)

    def open_plot_data_dialog(self):
        """
        Open a dialog to select data plotting options.
        """
        if not self.is_data_loaded:
            QMessageBox.warning(self, 'No Data Loaded', 'Please load data first.')
            return

        dialog = QDialog(self)
        dialog.setWindowTitle('Select Data Plotting Option')
        layout = QVBoxLayout()

        # Create a dropdown for data plotting options
        dropdown = QComboBox()
        dropdown.addItems(['data', 'data+ratio', 'eufspec+delchi'])
        dropdown.setCurrentText(self.data_plot_option if hasattr(self, 'data_plot_option') else 'data')

        # Connect dropdown selection to update the attribute
        dropdown.currentTextChanged.connect(lambda text: setattr(self, 'data_plot_option', text))

        # Add OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: [dialog.accept(), self.plot_data()])
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addWidget(dropdown)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        dialog.exec_()

    def plot_data(self):
        """
        Plot the data based on the selected option.
        """
        if not hasattr(self, 'data_plot_option'):
            self.data_plot_option = 'data'

        # Ensure plot configuration includes background
        Plot.background = self.include_background  # Set background plot visibility

        # Define arrays for plotting
        xs, ys, xerrs, yerrs, backs, mod_total, ratios, ratio_errors = [], [], [], [], [], [], [], []
        unf_xs, unf_ys, unf_xerrs, unf_yerrs, unf_model_total = [], [], [], [], []
        delchi, delchi_errors = [], []

        # Load data for plotting
        for i in range(AllData.nSpectra):
            Plot('ldata')
            # X-axis data
            xs.append(np.array(Plot.x(plotGroup=i+1)))
            # Y-axis data
            ys.append(np.array(Plot.y(plotGroup=i+1)))
            # X-axis errors
            xerrs.append(np.array(Plot.xErr(plotGroup=i+1)))
            # Y-axis errors
            yerrs.append(np.array(Plot.yErr(plotGroup=i+1)))
            try:
                # Background data
                backs.append(np.array(Plot.backgroundVals(plotGroup=i+1)))
            except Exception as e:
                print(f"Warning: {e}")
                backs.append(None)  # Append None if background data is not available
            # Model data
            mod_total.append(np.array(Plot.model(i+1)))

            Plot('ratio')
            ratios.append(np.array(Plot.y(plotGroup=i+1)))
            ratio_errors.append(np.array(Plot.yErr(plotGroup=i+1)))

            Plot('eufspec')
            unf_xs.append(np.array(Plot.x(plotGroup=i+1)))
            unf_ys.append(np.array(Plot.y(plotGroup=i+1)))
            unf_xerrs.append(np.array(Plot.xErr(plotGroup=i+1)))
            unf_yerrs.append(np.array(Plot.yErr(plotGroup=i+1)))
            unf_model_total.append(np.array(Plot.model(i+1)))

            Plot('delchi')
            delchi.append(np.array(Plot.y(plotGroup=i+1)))
            delchi_errors.append(np.array(Plot.yErr(plotGroup=i+1)))

        # Create a new figure and axis
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.add_subplot(111)
        if self.data_plot_option == 'data':
            for i in range(AllData.nSpectra):
                # Plot spectrum data with error bars
                self.ax.errorbar(xs[i], ys[i], xerr=xerrs[i], yerr=yerrs[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                # Plot model data
                self.ax.plot(xs[i], mod_total[i], label=f'Model {i+1}', color=SPECTRUM_COLORS[i])
                # Plot background data if available
                # Set plot scales and labels
                self.ax.set_xscale('log')  # Set x-axis to logarithmic scale
                self.ax.set_yscale('log')  # Set y-axis to logarithmic scale
                self.ax.set_xlabel('Energy (keV)')  # Label x-axis
                self.ax.set_ylabel('Counts s$^{-1}$ keV$^{-1}$ cm$^{-2}$')  # Label y-axis
                self.ax.legend()  # Add legend to plot
                self.ax.set_title(f"Plot of {self.data_plot_option}")  # Set plot title

                if backs[i] is not None and self.include_background:
                    self.ax.scatter(xs[i], backs[i], marker='*', label=f'Background {i+1}', color=SPECTRUM_COLORS[i])


        elif self.data_plot_option == 'data+ratio':
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            self.canvas.figure = fig
            self.ax = [ax1, ax2]

            for i in range(AllData.nSpectra):
                ax1.errorbar(xs[i], ys[i], xerr=xerrs[i], yerr=yerrs[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                ax1.plot(xs[i], mod_total[i], label=f'Model {i+1}', color=SPECTRUM_COLORS[i])

                # Set plot scales and labels
                ax1.set_xscale('log')  # Set x-axis to logarithmic scale
                ax1.set_yscale('log')  # Set y-axis to logarithmic scale
                ax1.set_xlabel('Energy (keV)')  # Label x-axis
                ax1.set_ylabel('Counts s$^{-1}$ keV$^{-1}$ cm$^{-2}$')  # Label y-axis
                ax1.legend()  # Add legend to plot
                ax1.set_title(f"Plot of data")  # Set plot title

                ax2.errorbar(xs[i], ratios[i], xerr = xerrs[i], yerr=ratio_errors[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                ax2.set_xlabel('Energy (keV)')  # Label x-axis
                ax2.set_ylabel('Ratio')  # Label y-axis
                ax2.legend()  # Add legend to plot
                ax2.set_title(f"Plot of ratio")  # Set plot title

        elif self.data_plot_option == 'eufspec+delchi':
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            self.canvas.figure = fig
            self.ax = [ax1, ax2]

            for i in range(AllData.nSpectra):

                ax1.errorbar(unf_xs[i], unf_ys[i], xerr=unf_xerrs[i], yerr=unf_yerrs[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                ax1.plot(unf_xs[i], unf_model_total[i], label=f'Model {i+1}', color=SPECTRUM_COLORS[i])

                # Set plot scales and labels
                ax1.set_xscale('log')  # Set x-axis to logarithmic scale
                ax1.set_yscale('log')  # Set y-axis to logarithmic scale
                ax1.set_xlabel('Energy (keV)')  # Label x-axis
                ax1.set_ylabel('Counts s$^{-1}$ keV$^{-1}$ cm$^{-2}$')  # Label y-axis
                ax1.legend()  # Add legend to plot
                ax1.set_title(f"Plot of eufspec")  # Set plot title

                ax2.errorbar(unf_xs[i], delchi[i], yerr=delchi_errors[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                ax2.set_xlabel('Energy (keV)')  # Label x-axis
                ax2.set_ylabel('Delchi')  # Label y-axis
                ax2.legend()  # Add legend to plot
                ax2.set_title(f"Plot of delchi")  # Set plot title

        if isinstance(self.ax, list):
            ax1, ax2 = tuple(self.ax)
            if hasattr(self, 'freeze_axes_selected') and self.freeze_axes_selected:
                ax1.set_xlim(self.ax1.get_xlim())
                ax1.set_ylim(self.ax1.get_ylim())
                ax2.set_xlim(self.ax2.get_xlim())
                ax2.set_ylim(self.ax2.get_ylim())
            else:
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
        else:
            if hasattr(self, 'freeze_axes_selected') and self.freeze_axes_selected:
                self.ax.set_xlim(self.ax.get_xlim())
                self.ax.set_ylim(self.ax.get_ylim())
            else:
                self.ax.relim()
                self.ax.autoscale_view()

        self.canvas.draw()

    def rescale_plot(self):
        """
        Rescale the y-axis limits of the plot based on the checkbox state.
        """
        if isinstance(self.ax, list):
            ax = self.ax[0]
        else:
            ax = self.ax
        y_data = [line.get_ydata() for line in ax.get_lines()]
        max_y = max([max(y) for y in y_data if len(y) > 0])
        ax.set_ylim(PLOT_Y_MIN_FACTOR * max_y, PLOT_Y_MAX_FACTOR * max_y)
        self.canvas.draw()

    def load_plot_style(self):
        """
        Load a matplotlib style file and apply the configurations to the plots.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Plot Style File", "", "Style Files (*.mplstyle);;All Files (*)", options=options)
        if file_path:
            try:
                plt.style.use(file_path)
                QMessageBox.information(self, "Load Plot Style", f"Plot style loaded from {file_path}")
                self.update_plot()  # Redraw the plot with the new style
            except Exception as e:
                QMessageBox.warning(self, "Load Plot Style", f"Failed to load plot style: {str(e)}")


# Create the PyQt application, which can handle arguments in sys.argv
app = QApplication(sys.argv)
# Create an instance of the main window
window = MainWindow()
# Show the main window
window.show()
# Start the application's event loop
sys.exit(app.exec())

# Xset.save('filename.xcm',info='a') # save the configuration of parameters in .xcm file
# 'a' saves all, data and model