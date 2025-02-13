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
    QScrollArea  # Creates scroll area
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)  # Allows embedding Matplotlib plots in PyQt
from matplotlib.figure import Figure  # Used to create the Matplotlib figure
from xspec import *

# Define a list of colors for plotting spectra
SPECTRUM_COLORS = ['black', 'red', 'lime', 'blue', 'cyan', 'magenta', 'yellow', 'orange']

# Load the relxill models
AllModels.lmod('relxill')


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
        self.setWindowTitle("XSPEC Model")
        self.setGeometry(100, 100, 1000, 600)  # Set window dimensions
        self.model_name = None

        # Create a menubar
        menubar = self.menuBar()

        # Add 'File' menu
        file_menu = menubar.addMenu('File')
        load_model_xcm_action = QAction('Load Model as XCM', self)  # Renamed action
        save_plot_action = QAction('Save Plot', self)
        save_xcm_action = QAction('Save Parameters as XCM', self)  # New action
        load_data_xcm_action = QAction('Load Data as XCM', self)  # New action
        exit_action = QAction('Exit', self)
        file_menu.addAction(load_model_xcm_action)  # Updated action
        file_menu.addAction(save_plot_action)
        file_menu.addAction(save_xcm_action)  # Add new action
        file_menu.addAction(load_data_xcm_action)  # Add new action
        file_menu.addAction(exit_action)

        # Add 'View' menu
        view_menu = menubar.addMenu('View')
        freeze_axes_action = QAction('Freeze Axes', self, checkable=True)
        use_textboxes_action = QAction('Use Textboxes for Parameters', self, checkable=True)
        include_background_action = QAction('Include Background For Data', self, checkable=True)
        set_axes_limits_action = QAction('Set Axes Limits', self)
        plot_components_action = QAction('Plot Different Components', self)
        select_plot_action = QAction('Select What to Plot', self)
        view_menu.addAction(freeze_axes_action)
        view_menu.addAction(use_textboxes_action)
        view_menu.addAction(include_background_action)
        view_menu.addAction(set_axes_limits_action)
        view_menu.addAction(plot_components_action)
        view_menu.addAction(select_plot_action)
        set_axes_limits_action.triggered.connect(self.set_axes_limits)

        # Connect actions to methods (placeholders)
        load_model_xcm_action.triggered.connect(self.load_model_as_xcm)  # Updated connection
        save_plot_action.triggered.connect(self.save_plot)
        save_xcm_action.triggered.connect(self.save_parameters_as_xcm)  # Connect new action
        load_data_xcm_action.triggered.connect(self.load_data_as_xcm)  # Connect new action
        exit_action.triggered.connect(self.close)
        freeze_axes_action.triggered.connect(lambda: self.toggle_option('Freeze Axes'))
        use_textboxes_action.triggered.connect(lambda: self.toggle_option('Use Textboxes for Parameters'))
        include_background_action.triggered.connect(self.toggle_background_visibility)
        plot_components_action.triggered.connect(self.plot_different_components)
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
        self.model_textbox.setPlaceholderText("e.g., powerlaw")
        self.model_textbox.returnPressed.connect(self.load_model)
        left_panel.addWidget(self.model_label)
        left_panel.addWidget(self.model_textbox)

        # Button to generate plot
        self.plot_button = QPushButton("Generate Plot")
        self.plot_button.clicked.connect(self.update_plot)
        left_panel.addWidget(self.plot_button)

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
        self.loaded = False
        self.what_to_plot = 'model'

    def load_model(self):
        """
        Load the XSPEC model entered in the textbox, initialize UI elements (sliders or textboxes)
        for model parameters, and update the plot.
        """
        if not self.loaded:
            # Retrieve the model name from the textbox and initialize the model
            self.model_name = self.model_textbox.text().strip()
            self.model = Model(self.model_name)

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
            labels = []
            indices = []
            labels_in_comps = []
            i = 1

            # Extract component names from the model
            comps = self.model.componentNames
            for component in comps:
                # Get parameter names for each component
                labels_in_component = getattr(self.model, component).parameterNames
                labels_in_comps.append(labels_in_component)
                for par in labels_in_component:
                    # Add parameter to list if it's not frozen and not linked
                    if not getattr(getattr(self.model, component), par).frozen and getattr(getattr(self.model, component), par).link == '':
                        indices.append(i)
                        if par == 'norm':
                            labels.append(par + '_' + component[0])
                        else:
                            labels.append(par)
                        i += 1

            # Store extracted labels and indices
            self.labels = labels
            self.indices = indices
            self.labels_in_comps = labels_in_comps
            self.comps = comps

            # Initialize model parameters list
            self.model_params = []
            for i in range(len(labels_in_comps)):
                for j in range(len(labels_in_comps[i])):
                    param = getattr(getattr(self.model, comps[i]), labels_in_comps[i][j])
                    if not param.frozen and param.link == '':
                        self.model_params.append(param)

            counter = 0
            for i in range(len(labels_in_comps)):
                component_label = QLabel(f"Component: {comps[i]}")
                left_panel.insertWidget(counter * 2 + 1, component_label)
                counter += 1

                for j in range(len(labels_in_comps[i])):
                    param = getattr(getattr(self.model, comps[i]), labels_in_comps[i][j])
                    label = QLabel(f"{param.name}: {param.values[0]:.3f}")  # Display parameter value

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
        precision_factor = 1000  # Adjust this to increase slider precision
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

    def update_plot(self):
        """
        Generate the plot using the current XSPEC model parameters.
        """
        if not hasattr(self, 'model'):
            self.load_model()

        # Ensure the model is set for the plot
        Plot.device = '/null'  # Suppress plot output
        Plot.xAxis = 'keV'
        Plot.xLog = True
        Plot.yLog = True

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
        self.ax.set_ylabel("Counts s$^{-1}$ keV$^{-1}$ cm$^{-2}$")
        self.ax.set_title(f"Plot of {self.model_name} {self.what_to_plot}")
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        if AllData.nSpectra >= 1:
            for i in range(AllData.nSpectra):
                self.ax.errorbar(self.xs[i], self.ys[i], xerr=self.xerrs[i], yerr=self.yerrs[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
                self.ax.plot(self.xs[i], self.mod_total[i], label=f'Model {i+1}', color=SPECTRUM_COLORS[i])
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
        self.load_model()

    def toggle_background_visibility(self, checked):
        """
        Toggle the visibility of the background in the plot.

        Parameters:
        - checked (bool): Whether the background should be visible.
        """
        self.include_background = checked
        self.update_plot()

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
        self.loaded = True
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open XCM File", "", "XCM Files (*.xcm);;All Files (*)", options=options)
        if file_path:
            Xset.restore(file_path)
            self.model = AllModels(1)

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

        # Restore XSPEC settings from the selected XCM file
        Xset.restore(file_path)  # Load the XCM file
        # Fit.perform()  # Perform the fit

        # Check if models are loaded and update instance attributes
        if AllModels(1):
            self.model = AllModels(1)  # Set the model from loaded data
            self.model_name = AllModels(1).name  # Get the model name
            print(f"Loaded model: {self.model_name}")  # Print the loaded model name
            self.loaded = True
            self.load_model()  # Update sliders and labels with the loaded model
        else:
            print("No model loaded from the XCM file.")  # Print warning if no model is loaded

        # Ensure plot configuration includes background
        Plot.background = True  # Set background plot to visible

        # Define arrays for plotting
        xs, ys, xerrs, yerrs, backs, mod_total = [], [], [], [], [], []

        # Load data for plotting
        Plot('ldata')
        for i in range(AllData.nSpectra):
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

        # Plot data on the existing canvas
        self.ax.clear()  # Clear previous plot
        for i in range(AllData.nSpectra):
            # Plot spectrum data with error bars
            self.ax.errorbar(xs[i], ys[i], xerr=xerrs[i], yerr=yerrs[i], fmt='.', label=f'Spectrum {i+1}', color=SPECTRUM_COLORS[i])
            # Plot model data
            self.ax.plot(xs[i], mod_total[i], label=f'Model {i+1}', color=SPECTRUM_COLORS[i])
            # Plot background data if available
            if backs[i] is not None and self.include_background:
                self.ax.scatter(xs[i], backs[i], marker='*', label=f'Background {i+1}', color=SPECTRUM_COLORS[i])

        # Set plot scales and labels
        self.ax.set_xscale('log')  # Set x-axis to logarithmic scale
        self.ax.set_yscale('log')  # Set y-axis to logarithmic scale
        self.ax.set_xlabel('Energy (keV)')  # Label x-axis
        self.ax.set_ylabel('Counts s$^{-1}$ keV$^{-1}$ cm$^{-2}$')  # Label y-axis
        self.ax.legend()  # Add legend to plot
        self.ax.set_title(f"Plot of {self.model_name} model")  # Set plot title

        # Draw the updated plot
        self.canvas.draw()  # Refresh the canvas with the new plot

        # Store data for later use
        self.xs = xs
        self.ys = ys
        self.xerrs = xerrs
        self.yerrs = yerrs
        self.backs = backs
        self.mod_total = mod_total


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
        buttons.accepted.connect(lambda: self.apply_axes_limits(x_min_textbox.text(), x_max_textbox.text(), y_min_textbox.text(), y_max_textbox.text(), dialog))
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
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numbers for the axes limits.')


    def plot_different_components(self):
        """
        Plot the different components of the model separately.
        """
        if not hasattr(self, 'model'):
            QMessageBox.warning(self, 'No Model Loaded', 'Please load a model first.')
            return

        # Clear previous plot
        self.ax.clear()

        # Store original model parameters
        original_params = {}

        # Iterate over each component
        for component in self.comps:
            # Check if the component has a 'norm' attribute
            if hasattr(getattr(self.model, component), 'norm'):
                # Store original normalization value
                original_params[component] = getattr(self.model, component).norm

                # Set all other components' normalization to 0
                for other_component in self.comps:
                    if other_component != component and hasattr(getattr(self.model, other_component), 'norm'):
                        getattr(self.model, other_component).norm = 0

                # Plot the data
                Plot('model')
                x = Plot.x()
                y = Plot.model()

                # Plot the component
                self.ax.plot(x, y, label=f'Component {component}', color=SPECTRUM_COLORS[self.comps.index(component) % len(SPECTRUM_COLORS)])

                # Restore original normalization values
                for comp, norm in original_params.items():
                    getattr(self.model, comp).norm = norm

        # Set plot labels and title
        self.ax.set_xlabel('Energy (keV)')
        self.ax.set_ylabel('Counts s$^{-1}$ keV$^{-1}$ cm$^{-2}$')
        self.ax.set_title('Plot of Different Model Components')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.legend()

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