# Customer Churn Prediction

## Overview

This project aims to predict customer churn using machine learning models. The dataset consists of various customer attributes, and the goal is to build a model that identifies customers who are likely to leave.

## Steps to Run the Project (Using Colab/JupyterNotebook)

### 1. Download the Dataset(Churn_Data.csv) and the ipynb file(CustomerChurnPredictor.ipynb) from my GitHub repository

### 2. Upload the dataset on Colab/Jupyter Notebook

### 3. Run the ipynb file

## Steps to Run the Project(From Github)

### 1. Clone the Repository

git clone https://github.com/Sparshjaiswal3/Customer-churn-prediction.git cd Customer-churn-prediction

### 2. Set Up the Environment

Ensure you have Python 3.7+ installed. Then install the required dependencies:
pip install -r requirements.txt

### 3. Download the Dataset

Place the dataset (`Churn_Data.csv`) in the `data/` directory. If not available, download it from [source link].

### 4. Run the Preprocessing Script

python preprocess.py

This script:

- Cleans the data
- Encodes categorical features
- Handles missing values
- Scales numerical features

### 5. Train the Model

python train.py

This script:

- Splits data into training and testing sets
- Handles class imbalance using SMOTE
- Trains Logistic Regression, Decision Tree, and Random Forest models
- Saves the trained models

### 6. Evaluate the Model

python evaluate.py

This script:

- Generates classification reports
- Displays feature importance (Random Forest)
- Plots the ROC curve
- Shows the confusion matrix

### 7. Run in Jupyter Notebook (Optional)

jupyter notebook CustomerChurnPredictor.ipynb

## Project Structure

├── data/ │ ├── Churn_Data.csv # Raw dataset │ ├── Processed_Customer_Churn.csv # Preprocessed dataset │ ├── models/ │ ├── logistic_model.pkl # Trained Logistic Regression │ ├── dt_model.pkl # Trained Decision Tree │ ├── rf_model.pkl # Trained Random Forest │ ├── preprocess.py # Data preprocessing script ├── train.py # Model training script ├── evaluate.py # Model evaluation script ├── requirements.txt # Required dependencies ├── CustomerChurnPredictor.ipynb # Jupyter Notebook for the project └── README.md # Project documentation

## Dependencies

Ensure you have the following installed:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `imblearn`
- `jupyter`

You can install them using:

pip install -r requirements.txt

##Contributing

Feel free to fork the repository and submit pull requests.
