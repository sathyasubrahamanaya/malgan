# Mal-GAN: Synthetic Malware Generation & Detection

## Overview

This project demonstrates an end-to-end workflow for training and deploying a Malware Generative Adversarial Network (Mal-GAN) model for both synthetic malware generation and malware detection. It comprises:

- A **training pipeline** (in a VS Code–style Python script with cell markers) that preprocesses data, trains a Random Forest “black box” for malware detection, builds the Mal-GAN model (including a generator and a substitute detector), and tests the models.
- A **production-level Streamlit application** that loads the saved models and scaler, allowing users to either generate synthetic malware samples or detect malware from user-provided input (via CSV upload or manual entry).

## Project Structure

```plaintext
.
├── malgan_training.py          # VS Code–style training script with cell markers
├── app.py                      # Production-level Streamlit application
├── dataset_malwares.csv        # Training dataset (malware and benign samples)
├── dataset_test.csv            # Test dataset
├── generator.h5                # Saved generator model (after training)
├── substituteDetector.h5       # Saved substitute detector model
├── malGAN.h5                   # Saved Mal-GAN model
├── blackBox_file.pkl           # Saved Random Forest black box model
├── scaler.pkl                  # Saved MinMaxScaler (fitted on feature columns only)
├── column_order.pkl            # Saved list of feature names (in the order used during training)
└── README.md                   # This README file
```

## Requirements

- **Python 3.x**

### Python Packages

Install the following packages using pip:

```bash
pip install numpy pandas scikit-learn tensorflow streamlit
```

## Training the Model

### 1. Prepare the Datasets

- Place your training (`dataset_malwares.csv`) and test (`dataset_test.csv`) datasets in the project directory.
- The datasets must include a `Malware` column (with `1` for malware and `0` for benign samples) along with the corresponding features.
  
### 2. Preprocess and Train

Open `malgan_training.py` in Visual Studio Code. This script is organized into cells (using `#%%` markers) so you can run each part interactively. The training pipeline includes:

- **Data Preprocessing:**  
  - Drops non-feature columns (e.g., "Name").
  - Selects important features based on correlation with the target.
  - **Crucially, the `Malware` column is dropped before scaling.**  
    This ensures that the scaler (a `MinMaxScaler`) is fit only on the feature data.
  - The fitted scaler and the feature order (column names) are saved as `scaler.pkl` and `column_order.pkl`, respectively.

- **Black Box Training:**  
  A Random Forest classifier is trained on malware and benign samples to serve as the detection “black box.” The model is saved as `blackBox_file.pkl`.

- **Mal-GAN Model:**  
  The Mal-GAN is built by combining:
  - A **generator** model (to generate synthetic malware samples).
  - A **substitute detector** (which provides feedback to the generator).
  
  Both models are trained and then saved as `generator.h5`, `substituteDetector.h5`, and `malGAN.h5`.

### 3. Running the Training Script

Run the script cell-by-cell in VS Code's interactive mode. Once complete, ensure that the following files are generated:
- `generator.h5`
- `substituteDetector.h5`
- `malGAN.h5`
- `blackBox_file.pkl`
- `scaler.pkl`
- `column_order.pkl`

## Running the Streamlit Application

### 1. Place All Required Files

Ensure that the following files are in the same directory as `app.py`:
- `generator.h5`
- `blackBox_file.pkl`
- `scaler.pkl`
- `column_order.pkl`

### 2. Launch the App

Run the following command from your terminal:

```bash
streamlit run app.py
```

This will open the Streamlit app in your default web browser.

### 3. Using the App

The Streamlit app offers two modes (selectable from the sidebar):

- **Generate Synthetic Malware:**  
  Click the “Generate Synthetic Malware” option to generate a synthetic malware sample using random input vectors (both malware sample and noise) of the expected dimensionality (derived from the saved `column_order`).

- **Detect Malware:**  
  Users have two options:
  - **CSV Upload:**  
    Upload a CSV file containing raw feature data (each row must have the exact number of features as saved in `column_order.pkl`). The app will reorder the columns if necessary, scale the data using the loaded scaler (`scaler.pkl`), and output predictions (1 = Malware, 0 = Benign).
  - **Manual Entry:**  
    Manually enter a comma-separated list of feature values. The app checks that the vector length matches the expected number (e.g., 69 features), scales the input, and then displays the prediction.

## Important Considerations

- **Scaling:**  
  The MinMaxScaler is fit only on the input feature columns (the `Malware` target is dropped). Any new input must be scaled using the saved scaler (`scaler.pkl`).

- **Feature Order:**  
  The expected order of features is saved in `column_order.pkl`. Whether using CSV or manual entry, ensure the feature order matches this saved list.

- **Input Dimensionality:**  
  The model expects an input vector of a fixed size (e.g., 69 features). Adjust your data accordingly.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project is inspired by the concepts in "Mastering Mal-GANs: A Step-by-Step Guide to Building Your Own Mal-GAN Model". Special thanks to the community for supporting advances in machine learning for cybersecurity.
```

---

This README file provides a comprehensive guide to setting up, training, and running the Mal-GAN project along with detailed usage instructions. Customize any parts as needed for your specific project details.