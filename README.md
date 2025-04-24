# ChiByEye: XSPEC Model Visualizer 🎛️📈

ChiByEye is an interactive PyQt5-based graphical interface for exploring and visualizing XSPEC spectral models. Designed for researchers and students in astrophysics, this tool allows users to intuitively tweak model parameters, view spectral components, and perform fits — all with real-time visual feedback.

---

## 🚀 Features

- 🧠 Load XSPEC models manually or from `.xcm` files
- 🔧 Adjust model parameters using sliders or textboxes
- 📊 Visualize:
  - Complete spectral models
  - Individual model components
  - Ratio and residual plots
- 🎨 Customize plot styles and axes
- 🧮 Perform fits using XSPEC's backend
- 📤 Export plots and save model states

---

## 📦 Requirements

To use ChiByEye, you need:

- Python 3.x
- XSPEC (via HEASoft) and its `pyxspec` Python bindings
- Python packages:
  ```bash
  pip install numpy matplotlib pyqt5
  ```

> ⚠️ If you don’t have the `relxill` model installed, comment out line 38:  
> `AllModels.lmod('relxill')`

---

## 🛠️ How to Run

```bash
python xspec_plot_interface.py
```

---

## 🖥 Interface Overview

- **Home screen**: Input a model name (e.g., `powerlaw`) or load one from the menu.
- **Generate Plot**: Visualizes the current model based on parameters.
- **Plot Menu**:
  - View different XSPEC modes (`model`, `emodel`, `eemodel`)
  - Plot different components
  - Plot same model N times (vary one parameter across values)
- **Data Plotting**: Load `.xcm` files with spectra and choose to view:
  - `data`
  - `data + ratio`
  - `eufspec + delchi`
- **View Menu**: Toggle sliders/textboxes, show frozen parameters, and set custom axes.

---

## 📂 File Operations

- **File → Load Model as XCM**: Load XSPEC model file
- **File → Load Data as XCM**: Load model + data
- **File → Save Parameters as XCM**: Export current state
- **File → Save Plot**: Save current figure as PNG
- **File → Load Plot Style File**: Apply custom `.mplstyle`

---

## 🧪 Example

An example file `test_file.xcm` is provided for demonstration.

---

## 📬 Contact

Questions or feedback? Reach out to us!

- Paulo Silva — MIT  
- Paul Draghis — MIT
- Email: [pauloh@mit.edu](pauloh@mit.edu)

---

## 📄 License

This project is released under the MIT License.  
If using XSPEC or relxill in your research, please cite them appropriately.

---

## 📚 Citation

If this software aids your research, please cite:

```
Arnaud, K. A. (1996). XSPEC: The First Ten Years.
Astronomical Data Analysis Software and Systems V, ASP Conf. Series, 101, 17.
```
