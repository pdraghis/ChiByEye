import sys
import numpy as np
import time
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
from PyQt5.QtCore import Qt, QCoreApplication, QProcess, QTimer, QThread, pyqtSignal, QObject
import threading

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
PLOT_Y_MIN_FACTOR = 10**(-5)
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
Plot.xAxis = 'keV'
Xset.parallel.leven = 1 # set number of cpus used in fitting to be 1 by default

# continue fit without interrupting
Fit.query = "yes"

class FitWorker(QObject):
    # Worker object to perform the fit in a separate thread.
    # Emits 'finished' signal with an exception (if any) when done.
    finished = pyqtSignal(object)  # emits exception or None

    def __init__(self, stop_event=None):
        super().__init__()
        # Event used to signal when the fit should be stopped
        self.stop_event = stop_event or threading.Event()
        self.exception = None

    def run(self):
        # This method is executed in the worker thread.
        # It performs the fit and handles exceptions.
        try:
            time.sleep(5)
            Fit.perform()  # Replace with actual fitting logic, check self.stop_event periodically if possible
        except Exception as e:
            self.exception = e
        # Emit the finished signal with the exception (or None if successful)
        self.finished.emit(self.exception)

class FitDialog(QDialog):
    def __init__(self, stop_callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fitting in Progress")
        self.setModal(True)
        self.setFixedSize(300, 120)
        layout = QVBoxLayout()
        label = QLabel("XSPEC is performing the fit...\nYou may stop the fit at any time.")
        label.setWordWrap(True)
        layout.addWidget(label)
        self.stop_button = QPushButton("Stop Fit")
        self.stop_button.clicked.connect(stop_callback)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

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
        load_model_xcm_action = QAction('Load Model as XCM', self) 
        save_plot_action = QAction('Save Plot', self)
        save_xcm_action = QAction('Save Parameters as XCM', self)  
        load_data_xcm_action = QAction('Load Data as XCM', self)
        load_plot_style_action = QAction('Load Plot Style File', self)  
        exit_action = QAction('Exit', self)
        restart_action = QAction('Restart', self)  
        file_menu.addAction(load_model_xcm_action)  
        file_menu.addAction(save_plot_action)
        file_menu.addAction(save_xcm_action)  
        file_menu.addAction(load_data_xcm_action)  
        file_menu.addAction(load_plot_style_action)  
        file_menu.addAction(restart_action)  
        file_menu.addAction(exit_action)

        # Add 'View' menu
        view_menu = menubar.addMenu('View')
        freeze_axes_action = QAction('Freeze Axes', self, checkable=True)
        use_textboxes_action = QAction('Use Sliders for Parameters', self, checkable=True)
        set_axes_limits_action = QAction('Set Axes Limits', self)
        show_frozen_action = QAction('Show Frozen Parameters', self, checkable=True) 

        view_menu.addAction(freeze_axes_action)
        view_menu.addAction(use_textboxes_action)
        view_menu.addAction(set_axes_limits_action)
        view_menu.addAction(show_frozen_action) 

        # Connect actions to their respective methods
        freeze_axes_action.triggered.connect(lambda: self.toggle_option('Freeze Axes'))
        use_textboxes_action.triggered.connect(lambda: self.toggle_option('Use Sliders for Parameters'))
        set_axes_limits_action.triggered.connect(self.set_axes_limits)
        show_frozen_action.triggered.connect(lambda: self.toggle_option('Show Frozen Parameters')) 

        # Add 'Plot' menu
        plot_menu = menubar.addMenu('Plot')

        # Add actions to the 'Plot' menu
        plot_components_action = QAction('Plot Different Components', self, checkable=True)
        select_plot_action = QAction('Plot Model', self)
        plot_n_times_action = QAction('Plot Same Curve N Times', self)  
        plot_data_action = QAction('Plot Data', self)  
        include_background_action = QAction('Include Background For Data', self, checkable=True)
        plot_menu.addAction(plot_components_action)
        plot_menu.addAction(select_plot_action)
        plot_menu.addAction(plot_n_times_action)
        plot_menu.addAction(plot_data_action)
        plot_menu.addAction(include_background_action)

        # Add 'Fit' menu and add 'Perform Fit' action there
        fit_menu = menubar.addMenu('Fit')
        perform_fit_action = QAction('Perform Fit', self)
        fit_menu.addAction(perform_fit_action)

        # Add 'Set Number of CPUs' action to the Fit menu
        set_cpus_action = QAction('Set Number of CPUs', self)
        fit_menu.addAction(set_cpus_action)
        set_cpus_action.triggered.connect(self.open_set_cpus_dialog)

        # Connect actions to their respective methods
        plot_components_action.triggered.connect(lambda: self.toggle_option('Plot Different Components'))
        select_plot_action.triggered.connect(self.open_select_plot_dialog)
        plot_n_times_action.triggered.connect(self.plot_same_curve_n_times)
        plot_data_action.triggered.connect(self.open_plot_data_dialog)
        include_background_action.triggered.connect(lambda: self.toggle_option('Include Background'))
        # Connect the action to the perform_fit method
        perform_fit_action.triggered.connect(self.perform_fit_threaded)

        # Connect actions to methods (placeholders)
        load_model_xcm_action.triggered.connect(self.load_model_as_xcm)  
        save_plot_action.triggered.connect(self.save_plot)
        save_xcm_action.triggered.connect(self.save_parameters_as_xcm) 
        load_data_xcm_action.triggered.connect(self.load_data_as_xcm)  # Connect new action
        load_plot_style_action.triggered.connect(self.load_plot_style)  # Connect new action
        exit_action.triggered.connect(self.close)
        restart_action.triggered.connect(self.restart_application)  # Connect new action

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
        self.use_textboxes_selected = True
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
            try:
                self.models = [Model(self.model_name)]
            except Exception as e:
                QMessageBox.warning(self, "Invalid Model", f"Could not load model '{self.model_name}'.\nError: {str(e)}")
                return  # Exit early if model is invalid

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

            for model in reversed(self.models):
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
                                textbox.editingFinished.connect(lambda tb=textbox, p=param, l=label: (setattr(p, 'values', [float(tb.text())]), l.setText(f"{p.name}: {p.values[0]:.3f}")))
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
            if self.plot_components_selected:
                self.plot_different_components()
            elif self.is_data_loaded:
                self.plot_data()
            else:
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
        if self.plot_components_selected:
            self.plot_different_components()
        elif self.is_data_loaded:
            self.plot_data()
        else:
            # Ensure the model is set for the plot
            if not hasattr(self, 'models'):
                self.load_model()
            if hasattr(self, 'models'): # Check that loading the model worked
                self.update_plot()

    def update_plot(self, plot_model=False):
        """
        Generate the plot using the current XSPEC model parameters.
        """
        Plot.device = '/null'  # Suppress plot output
        Plot.xLog = PLOT_X_LOG
        Plot.yLog = PLOT_Y_LOG

        if AllData.nSpectra > 0 and not plot_model:
            Plot("ldata")
        else:
            Plot(self.what_to_plot)  # Load the model

        # Extract data from XSPEC plot
        x = Plot.x()
        y = Plot.model()

        # Store current axes limits
        if isinstance(self.ax, list):
            ax = self.ax[0]
        else:
            ax = self.ax
        
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        self.canvas.figure.clear()
        self.ax = self.canvas.figure.add_subplot(111)

        # Generate the plot
        self.ax.plot(x, y, label=self.what_to_plot)
        self.ax.set_xlabel("Energy (keV)")
        if self.what_to_plot == "model":
            self.ax.set_ylabel(r"$\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1}$")
        elif self.what_to_plot == "emodel":
            self.ax.set_ylabel(r"$\rm keV\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
        elif self.what_to_plot == "eemodel":
            self.ax.set_ylabel(r"$\rm keV^2\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        if AllData.nSpectra >= 1 and not plot_model:
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
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.canvas.updateGeometry()
        # Optionally, force resize if needed:
        # if self.canvas.parent() is not None:
        #     self.canvas.resize(self.canvas.parent().size())

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
        else:
            if option_name == 'Show Frozen Parameters':
                self.show_frozen_parameters = not getattr(self, 'show_frozen_parameters', False)
            elif option_name == 'Use Sliders for Parameters':
                self.use_textboxes_selected = not getattr(self, 'use_textboxes_selected', False)
            elif option_name == 'Plot Different Components':
                self.plot_components_selected = not getattr(self, 'plot_components_selected', False)
            elif option_name == 'Include Background':
                self.include_background = not getattr(self, 'include_background', False)
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

        # Initialize data attributes
        self.xs, self.ys, self.xerrs, self.yerrs, self.backs = [], [], [], [], []
        self.ratios, self.ratio_errors = [], []
        self.unf_xs, self.unf_ys, self.unf_xerrs, self.unf_yerrs = [], [], [], []
        self.delchi, self.delchi_errors = [], []

        # Load data for plotting
        for i in range(AllData.nSpectra):
            Plot('ldata')
            self.xs.append(np.array(Plot.x(plotGroup=i+1)))
            self.ys.append(np.array(Plot.y(plotGroup=i+1)))
            self.xerrs.append(np.array(Plot.xErr(plotGroup=i+1)))
            self.yerrs.append(np.array(Plot.yErr(plotGroup=i+1)))
            try:
                self.backs.append(np.array(Plot.backgroundVals(plotGroup=i+1)))
            except Exception as e:
                print(f"Warning: {e}")
                self.backs.append(None)  # Append None if background data is not available

            Plot('ratio')
            self.ratios.append(np.array(Plot.y(plotGroup=i+1)))
            self.ratio_errors.append(np.array(Plot.yErr(plotGroup=i+1)))

            Plot('eufspec')
            self.unf_xs.append(np.array(Plot.x(plotGroup=i+1)))
            self.unf_ys.append(np.array(Plot.y(plotGroup=i+1)))
            self.unf_xerrs.append(np.array(Plot.xErr(plotGroup=i+1)))
            self.unf_yerrs.append(np.array(Plot.yErr(plotGroup=i+1)))

            Plot('delchi')
            self.delchi.append(np.array(Plot.y(plotGroup=i+1)))
            self.delchi_errors.append(np.array(Plot.yErr(plotGroup=i+1)))

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
            if isinstance(self.ax, list):
                ax1, ax2 = tuple(self.ax)
                if x_min:
                    ax1.set_xlim(left=float(x_min))
                    ax2.set_xlim(left=float(x_min))
                if x_max:
                    ax1.set_xlim(right=float(x_max))
                    ax2.set_xlim(right=float(x_max))
                if y_min:
                    ax1.set_ylim(bottom=float(y_min))
                if y_max:
                    ax1.set_ylim(top=float(y_max))
            else:
                if x_min:
                    self.ax.set_xlim(left=float(x_min))
                if x_max:
                    self.ax.set_xlim(right=float(x_max))
                if y_min:
                    self.ax.set_ylim(bottom=float(y_min))
                if y_max:
                    self.ax.set_ylim(top=float(y_max))
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            self.canvas.updateGeometry()
            # Optionally, force resize if needed:
            # if self.canvas.parent() is not None:
            #     self.canvas.resize(self.canvas.parent().size())
            dialog.accept()
            self.freeze_axes_selected = True
            AllModels.setEnergies(f'{x_min} {x_max} 1000 log')
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
        if isinstance(self.ax, list):
            ax = self.ax[0]
        else:
            ax = self.ax
        
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        self.canvas.figure.clear()
        self.ax = self.canvas.figure.add_subplot(111)


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
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.canvas.updateGeometry()
        # Optionally, force resize if needed:
        # if self.canvas.parent() is not None:
        #     self.canvas.resize(self.canvas.parent().size())

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
        buttons.accepted.connect(lambda: [dialog.accept(), self.update_plot(True)])
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addWidget(dropdown)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        dialog.exec_()

    def plot_same_curve_n_times(self):
        """
        Display a dialog to input parameters and plot the same curve multiple times for a selected model.
        """
        if not hasattr(self, 'models') or not self.models:
            QMessageBox.warning(self, 'No Models Loaded', 'Please load models first.')
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Plot Same Curve N Times")

        layout = QVBoxLayout()

        # Dropdown for selecting the model
        model_label = QLabel("Select model:")
        model_combo = QComboBox()
        model_combo.addItems([model.name for model in self.models])
        # layout.addWidget(model_label)
        # layout.addWidget(model_combo)

        # Dropdown for selecting parameter
        param_label = QLabel("Select parameter:")
        param_combo = QComboBox()
        layout.addWidget(param_label)
        layout.addWidget(param_combo)

        # Update parameter dropdown when a model is selected
        def update_param_dropdown():
            selected_model = self.models[model_combo.currentIndex()]
            param_combo.clear()
            for component in selected_model.componentNames:
                for param in getattr(selected_model, component).parameterNames:
                    param_combo.addItem(f"{param} ({component})", (component, param))

        model_combo.currentIndexChanged.connect(update_param_dropdown)
        update_param_dropdown()  # Initialize the parameter dropdown

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

        # Clear previous plot
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.add_subplot(111)

        def on_accept():
            selected_model = self.models[model_combo.currentIndex()]
            component, param_name = param_combo.currentData()
            param_obj = getattr(getattr(selected_model, component), param_name)

            # Get hard min and max from XSPEC parameter
            hard_min = param_obj.values[2]
            hard_max = param_obj.values[5]

            try:
                param_min = float(param_min_input.text())
                param_max = float(param_max_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Parameter min and max must be numbers.")
                return

            # Check that min < max and both are within hard limits
            if not (hard_min <= param_min < param_max <= hard_max):
                QMessageBox.warning(
                    self,
                    "Invalid Parameter Range",
                    f"Parameter values must satisfy:\n"
                    f"{hard_min} ≤ min < max ≤ {hard_max}"
                )
                return

            try:
                n = int(n_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Number of curves must be an integer.")
                return
            spacing = spacing_combo.currentText()
            cmap_name = cmap_combo.currentText()

            # Get the parameter object
            param_obj = getattr(getattr(selected_model, component), param_name)

            # Plot the loaded data if available
            if AllData.nSpectra >= 1:
                for i in range(AllData.nSpectra):
                    self.ax.errorbar(self.xs[i], self.ys[i], xerr=self.xerrs[i], yerr=self.yerrs[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                    if self.backs[i] is not None and self.include_background:
                        self.ax.scatter(self.xs[i], self.backs[i], marker='*', label=f'Background {i+1}', color=SPECTRUM_COLORS[i])
            
            # Determine color normalization based on spacing
            if spacing == "linear":
                norm = mcolors.Normalize(vmin=param_min, vmax=param_max)
            else:
                norm = mcolors.LogNorm(vmin=param_min, vmax=param_max)

            # Generate colors
            values = np.linspace(param_min, param_max, n)
            cmap = cm.get_cmap(cmap_name)
            colors = cmap(norm(values))

            # Store the original parameter value
            original_value = param_obj.values[0]

            # Determine the full energy range from all loaded spectra
            if hasattr(self, 'xs') and len(self.xs) > 0:
                emin = float(np.min([np.min(x) for x in self.xs if len(x)>0]))
                emax = float(np.max([np.max(x) for x in self.xs if len(x)>0]))
                # reset the model energy grid to cover every channel of every spectrum
                AllModels.setEnergies(f"{emin} {emax} 1000 log")
            
            # Plot the same curve N times with specified spacing and colors
            for i in range(n):
                # Adjust the model parameter
                if spacing == "linear":
                    param_value = param_min + i * (param_max - param_min) / (n - 1)
                else:
                    param_value = param_min * (param_max / param_min) ** (i / (n - 1))

                param_obj.values = [param_value] + param_obj.values[1:]

                # Recalculate the folded model on the new energy grid
                Plot("model")
                x = Plot.x()       # full energy axis from emin→emax
                y = Plot.model()   # model evaluation at that grid
                # Plot the curve
                self.ax.plot(x, y, label=f'{param_name}={param_value:.2f}', linestyle='--', color=colors[i])

            # Restore the original parameter value
            param_obj.values = [original_value] + param_obj.values[1:]

            # Set plot labels and title
            self.ax.set_xlabel('Energy (keV)')
            if self.what_to_plot == "model":
                self.ax.set_ylabel(r"$\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1}$")
            elif self.what_to_plot == "emodel":
                self.ax.set_ylabel(r"$\rm keV\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
            elif self.what_to_plot == "eemodel":
                self.ax.set_ylabel(r"$\rm keV^2\;(\rm Photons\;cm^{-2}\;s^{-1}\;keV^{-1})$")
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
            self.ax.legend()

            # Refresh canvas
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            self.canvas.updateGeometry()
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

        if isinstance(self.ax, list):
            x_limit = self.ax[0].get_xlim()
            y_limit = self.ax[0].get_ylim()
        else:
            x_limit = self.ax.get_xlim()
            y_limit = self.ax.get_ylim()

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


        Plot('ldata')
        chisq = str(round(Fit.statistic,2))+'/'+str(Fit.dof)
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

                if backs[i] is not None and self.include_background:
                    self.ax.scatter(xs[i], backs[i], marker='*', label=f'Background {i+1}', color=SPECTRUM_COLORS[i])


        elif self.data_plot_option == 'data+ratio':
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            # Replace the canvas with a new FigureCanvas
            new_canvas = FigureCanvas(fig)
            layout = self.centralWidget().layout()
            layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas = new_canvas
            layout.addWidget(self.canvas, stretch=3)
            self.ax = [ax1, ax2]
            ax2.axhline(y=1, c='lime', lw=2, zorder=-100)

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
                ax2.set_title(r'$\chi^2/\nu='+ chisq + '$')  # Set plot title

        elif self.data_plot_option == 'eufspec+delchi':
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            # Replace the canvas with a new FigureCanvas
            new_canvas = FigureCanvas(fig)
            layout = self.centralWidget().layout()
            layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas = new_canvas
            layout.addWidget(self.canvas, stretch=3)
            self.ax = [ax1, ax2]
            ax2.axhline(y=0, c='lime', lw=2, zorder=-100)

            for i in range(AllData.nSpectra):

                ax1.errorbar(unf_xs[i], unf_ys[i], xerr=unf_xerrs[i], yerr=unf_yerrs[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                ax1.plot(unf_xs[i], unf_model_total[i], label=f'Model {i+1}', color=SPECTRUM_COLORS[i])

                # Set plot scales and labels
                ax1.set_xscale('log')  # Set x-axis to logarithmic scale
                ax1.set_yscale('log')  # Set y-axis to logarithmic scale
                ax1.set_xlabel('Energy (keV)')  # Label x-axis
                ax1.set_ylabel('keV (Photons s$^{-1}$ keV$^{-1}$ cm$^{-2})$')  # Label y-axis
                ax1.legend()  # Add legend to plot
                ax1.set_title("Plot of eufspec")  # Set plot title

                ax2.errorbar(unf_xs[i], delchi[i], yerr=delchi_errors[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                ax2.set_xlabel('Energy (keV)')  # Label x-axis
                ax2.set_ylabel('Delchi')  # Label y-axis
                ax2.set_title("Plot of delchi")  # Set plot title


        if isinstance(self.ax, list):
            ax1, ax2 = tuple(self.ax)
            if hasattr(self, 'freeze_axes_selected') and self.freeze_axes_selected:
                ax1.set_xlim(x_limit)
                ax1.set_ylim(y_limit)
                ax2.set_xlim(x_limit)
                ax2.set_ylim(y_limit)
            elif self.rescale_checkbox.isChecked():
                self.rescale_plot()
            else:    
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
        else:
            if hasattr(self, 'freeze_axes_selected') and self.freeze_axes_selected:
                self.ax.set_xlim(x_limit)
                self.ax.set_ylim(y_limit)
            elif self.rescale_checkbox.isChecked():
                self.rescale_plot()
            else:
                self.ax.relim()
                self.ax.autoscale_view()

        

        self.xs = xs
        self.ys = ys
        self.xerrs = xerrs
        self.yerrs = yerrs

        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.canvas.updateGeometry()
        # Optionally, force resize if needed:
        # if self.canvas.parent() is not None:
        #     self.canvas.resize(self.canvas.parent().size())

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
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.canvas.updateGeometry()
        # Optionally, force resize if needed:
        # if self.canvas.parent() is not None:
        #     self.canvas.resize(self.canvas.parent().size())

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

    def perform_fit_threaded(self):
        """
        Starts the fitting process in a separate thread using QThread and FitWorker.
        This prevents the GUI from freezing during long-running fits and allows the user to stop the fit.
        Shows a modal dialog with a stop button while the fit is running.
        """
        # Create an event to signal the worker to stop if requested
        self.fit_stop_event = threading.Event()
        # Create a new QThread instance
        self.fit_thread = QThread()
        # Create the worker that will perform the fit, passing the stop event
        self.fit_worker = FitWorker(stop_event=self.fit_stop_event)
        # Move the worker to the thread (so its run method executes in the new thread)
        self.fit_worker.moveToThread(self.fit_thread)
        # When the thread starts, call the worker's run method
        self.fit_thread.started.connect(self.fit_worker.run)
        # When the worker finishes, handle results and clean up
        self.fit_worker.finished.connect(self.fit_finished)
        self.fit_worker.finished.connect(self.fit_thread.quit)
        self.fit_worker.finished.connect(self.fit_worker.deleteLater)
        self.fit_thread.finished.connect(self.fit_thread.deleteLater)
        # Create and show a dialog to allow the user to stop the fit
        self.fit_dialog = FitDialog(stop_callback=self.stop_fit, parent=self)
        # Start the fit thread
        self.fit_thread.start()
        self.fit_dialog.show()

    def stop_fit(self):
        # Called when the user clicks the stop button in the dialog.
        # Signals the worker thread to stop by setting the event.
        if hasattr(self, 'fit_stop_event'):
            self.fit_stop_event.set()
        # Set a flag to indicate the fit was stopped by the user
        self.fit_was_stopped = True
        # Close the dialog window
        if hasattr(self, 'fit_dialog'):
            self.fit_dialog.close()
        # Inform the user that the fit was stopped
        QMessageBox.information(self, "Fit Stopped", "The fit was stopped by the user.")

    def fit_finished(self, exception):
        # Called when the worker signals that the fit is finished (success or failure).
        # Ensures the dialog is closed and handles the result or any exception.
        if hasattr(self, 'fit_dialog'):
            self.fit_dialog.close()
        # Only update the plot if the fit was not stopped by the user
        if getattr(self, 'fit_was_stopped', False):
            self.fit_was_stopped = False  # Reset for next fit
            return
        if exception is None:
            # Fit completed successfully
            QMessageBox.information(self, "Fit Performed", "The fit has been successfully performed.")
            self.plot_data()  # Update the plot with the new fit results
        else:
            # An error occurred during fitting
            QMessageBox.warning(self, "Fit Error", f"An error occurred while performing the fit: {str(exception)}")

    def open_set_cpus_dialog(self):
        """
        Open a dialog allowing the user to select the number of CPUs for parallel fitting.
        Sets Xset.parallel.leven accordingly.
        """
        from PyQt5.QtWidgets import QInputDialog
        # Get current value if available, else default to 1
        current_cpus = getattr(Xset.parallel, 'leven', 1)
        num, ok = QInputDialog.getInt(self, "Set Number of CPUs", "Number of CPUs to use for parallel fitting:", value=current_cpus, min=1, max=64)
        if ok:
            Xset.parallel.leven = num
            QMessageBox.information(self, "Set Number of CPUs", f"XSPEC will now use {num} CPU(s) for parallel fitting.")


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