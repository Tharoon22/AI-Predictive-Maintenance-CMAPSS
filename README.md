# AI-Based Predictive Maintenance Using Machine Learning

Predicting **Remaining Useful Life (RUL)** of turbofan engines using machine learning and the **NASA C-MAPSS FD001** dataset.

## 📌 Project Overview

Predictive maintenance aims to estimate the remaining useful life of a machine before failure occurs. This project uses sensor data from turbofan engines to train a **Random Forest Regression** model that predicts the remaining operating cycles of an engine.

The project focuses on data preprocessing, RUL calculation, sensor-based feature engineering, machine learning, and evaluation on the provided test dataset.

## 📊 Dataset

**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Dataset
**Subset:** FD001

FD001 contains run-to-failure turbofan engine data with multiple sensor measurements recorded over successive operating cycles.

The project uses:

* Training engine data
* Test engine data
* Ground-truth RUL values for the test engines
* Multiple sensor measurements recorded over engine cycles

## ⚙️ Methodology

The project follows these main steps:

1. Load and preprocess the CMAPSS FD001 dataset.
2. Calculate RUL for the training data from the maximum cycle of each engine.
3. Calculate RUL for the test data using the provided ground-truth RUL values.
4. Select relevant sensor measurements.
5. Generate rolling-window features using a 5-cycle window.
6. Create rolling mean and rolling standard deviation features.
7. Cap RUL values at 125 cycles.
8. Normalize the input features using MinMaxScaler.
9. Train a Random Forest Regressor.
10. Evaluate predictions using RMSE, MAE and R².
11. Analyze feature importance.
12. Track predicted RUL for an individual engine over its operating cycles.

## 🧮 Feature Engineering

The model uses **13 selected sensor channels**.

For each selected sensor, three types of features are used:

* Original sensor value
* 5-cycle rolling mean
* 5-cycle rolling standard deviation

This results in:

**39 input features in total.**

The rolling features are calculated separately for each engine so that the temporal behavior of the sensor measurements is retained.

## 🤖 Machine Learning Model

A **Random Forest Regressor** is used for RUL prediction.

### Model configuration

| Parameter                |      Value |
| ------------------------ | ---------: |
| Number of trees          |        150 |
| Maximum depth            |         15 |
| Minimum samples per leaf |          2 |
| Random seed              |         42 |
| RUL cap                  | 125 cycles |
| Rolling window           |   5 cycles |

The model is trained using the processed training data and evaluated on the actual CMAPSS FD001 test set.

## 📈 Results

The final model achieved approximately:

| Metric |           Result |
| ------ | ---------------: |
| RMSE   | **16.83 cycles** |
| MAE    | **11.74 cycles** |
| R²     |         **0.63** |

The model provides a reasonable prediction of engine degradation and demonstrates how machine learning can be applied to condition-based and predictive maintenance.

## 📊 Visualizations

The project includes visualizations for:

### Actual vs Predicted RUL

Shows the relationship between the actual remaining useful life and the RUL predicted by the Random Forest model.

![Actual vs Predicted RUL](results/Phase-5%20Random%20Forest%20-%20Actual%20vs%20Predicted%20RUL.png)

### Feature Importance

Identifies the sensor-derived features that have the greatest influence on the model's predictions.

![Top 15 Feature Importance](results/Phase-5%20Top%2015%20features%20driving%20model%20predictions.png)

### Engine RUL Tracking

Tracks the actual and predicted RUL of an individual test engine over its operating cycles.

![Engine Unit-5 RUL Tracking](results/Phase-5%20Engine%20Unit-5%20RUL%20tracking%20over%20time.png)

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Random Forest Regression

## 📁 Project Structure

```text
AI-Predictive-Maintenance-CMAPSS/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/
│   └── predictive_maintenance.py
│
├── results/
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── rul_tracking.png
│
└── models/
    ├── rf_model.pkl
    └── scaler.pkl
```

## ▶️ How to Run

1. Download the NASA C-MAPSS FD001 dataset.
2. Place the required dataset files in a local data directory.
3. Update the `DATA_PATH` variable in the Python script to point to the dataset location.
4. Install the required Python packages:

```bash
pip install -r requirements.txt
```

5. Run:

```bash
python src/predictive_maintenance.py
```

The trained model, scaler and generated plots will be saved after execution.

## ⚠️ Limitations

* The project uses only the FD001 subset of the C-MAPSS dataset.
* The model is trained and evaluated on a simulated turbofan degradation dataset rather than live engine data.
* Random Forest does not explicitly model long-term temporal dependencies.
* The current implementation is intended as a project-level demonstration rather than a production predictive-maintenance system.

## 🚀 Future Improvements

Possible improvements include:

* Hyperparameter tuning
* XGBoost or LightGBM comparison
* LSTM-based time-series modelling
* SHAP-based model interpretability
* More C-MAPSS subsets such as FD002, FD003 and FD004
* Integration with real-time engine sensor data
* OBD-II/CAN-based data acquisition
* Deployment as a real-time predictive maintenance system

## 👤 Author

**Tharoon K**
B.E. Automobile Engineering
Madras Institute of Technology, Anna University
