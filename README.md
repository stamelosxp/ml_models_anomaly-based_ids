# Machine Learning Models for Network Intrusion Detection: Binary and Multi-Class Classification

This repository contains implementations of various machine learning and deep learning models for binary and multi-class classification tasks. The models are trained, evaluated, and compared using metrics such as accuracy, recall, precision, F1-score, ROC AUC, and log loss. Additionally, SHAP (SHapley Additive exPlanations) is used for feature importance analysis.    
## Dataset Used    
The dataset used in this project is the [CSE-CIC-IDS2018 Cleaned Dataset](https://www.kaggle.com/datasets/ekkykharismadhany/csecicids2018-cleaned/data), a cleaned and preprocessed version of the CSE-CIC-IDS2018 dataset, designed for intrusion detection system (IDS) research and classification of cybersecurity attacks.    

---
## Table of Contents    
- [Features](#features)    
- [Models Implemented](#models-implemented)    
- [Installation](#installation)    
- [Usage](#usage)    
- [Project Structure](#project-structure)    
- [Dependencies](#dependencies)    
- [Contributors](#contributors)
- [License](#license)    

---
## Features    
- Binary Classification: Models to distinguish between benign and attack classes.    
- Multi-Class Classification: Models to identify multiple types of attacks.    
- Hyperparameter Tuning: Automated hyperparameter optimization using Optuna and GridSearchCV.    
- Feature Importance: SHAP analysis for feature selection and interpretability.    
- Cross-Validation: K-Fold cross-validation for model evaluation.    
- Visualization: Confusion matrices, performance metrics, and SHAP plots for model interpretability.    

---
## Models Implemented    
### Binary Classification    
- XGBoost: Gradient boosting algorithm for binary classification.    
- MLP (Multi-Layer Perceptron): Neural network model for binary classification.    
- Isolation Forest: Anomaly detection model adapted for binary classification.    
### Multi-Class Classification    
- XGBoost: Gradient boosting algorithm for multi-class classification.    
- MLP (Multi-Layer Perceptron): Neural network model for multi-class classification.  

---
## Installation    
1. Clone the repository:    
```bash
git clone https://github.com/your-username/your-repo-name.gitcd your-repo-name
```
2. Create and activate a virtual environment:    
```bash
python -m venv venvsource venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required packages:    
```bash  
pip install -r requirements.txt
```

---
## Usage  

### 1. Preprocessing  
  
Ensure the following preprocessed datasets are available in the root directory:  
  
- `X_train_bin.csv`, `X_test_bin.csv`  
- `X_train_multi.csv`, `X_test_multi.csv`
- `y_train_bin.csv`, `y_test_bin.csv`  
- `y_train_multi.csv`, `y_test_multi.csv` 
  
These files are excluded from version control and must be generated or obtained separately.  
  
### 2. Running Models  

#### Binary Classification  
  
- **XGBoost**:  
  - `xgb_bin_hyper_tun.ipynb`  
  - `xgb_bin_best.ipynb`  
  
- **MLP**:  
  - `mlp_bin_hyper_tun.ipynb`  
  - `mlp_bin_best.ipynb`  
  
- **Isolation Forest**:  
  - `iso_hyper_tun.ipynb`  
  - `iso_best.ipynb`  
  
#### Multi-Class Classification  
  
- **XGBoost**:  
  - `xgb_multi_hyper_tun.ipynb`  
  - `xgb_multi_best.ipynb`  
  
- **MLP**:  
  - `mlp_multi_hyper_tun.ipynb`  
  - `mlp_multi_best.ipynb`  
  
### 3. Results Visualization  
  
Use `plot_results.ipynb` to generate visual comparisons of model performance including metrics and SHAP-based feature importance plots.  

---
## Project Structure  

The repository is organized as follows:  

- **`preprocessing.ipynb`**: Data preprocessing including cleaning, encoding, and feature selection.
- **`xgb_bin_hyper_tun.ipynb`**: Hyperparameter tuning for XGBoost binary classification.  
- **`xgb_bin_best.ipynb`**: Best XGBoost model for binary classification.  
- **`mlp_bin_hyper_tun.ipynb`**: Hyperparameter tuning for MLP binary classification.  
- **`mlp_bin_best.ipynb`**: Best MLP model for binary classification.  
- **`iso_hyper_tun.ipynb`**: Hyperparameter tuning for Isolation Forest.  
- **`iso_best.ipynb`**: Best Isolation Forest model for binary classification.  
- **`xgb_multi_hyper_tun.ipynb`**: Hyperparameter tuning for XGBoost multi-class classification.  
- **`xgb_multi_best.ipynb`**: Best XGBoost model for multi-class classification.  
- **`mlp_multi_hyper_tun.ipynb`**: Hyperparameter tuning for MLP multi-class classification.  
- **`mlp_multi_best.ipynb`**: Best MLP model for multi-class classification.  
- **`plot_results.ipynb`**: Visualization of model performance metrics and feature importance.  
- **`requirements.txt`**: List of dependencies required to run the project.  

---  
## Dependencies  
  
The project requires the following Python libraries:  
  
- `numpy`  
- `pandas`  
- `scikit-learn`  
- `matplotlib`  
- `seaborn`  
- `torch`  
- `xgboost`  
- `shap`  
- `optuna`  
  
Install all dependencies using the `requirements.txt` file.  

---
## Contributors  

- **Voulgaris Nikolaos** - [GitHub](https://github.com/NickVoulg02)   
- **Stamelos Charilaos-Panagiotis** - [GitHub](https://github.com/stamelosxp)

---
## License  
  
This project is licensed under the MIT License.
  


