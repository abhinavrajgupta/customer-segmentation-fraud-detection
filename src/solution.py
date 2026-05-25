
pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============== HELPER FUNCTIONS ===============

def print_section(title):
    # print a clean section header
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_metrics(y_true, y_pred, y_proba=None, model_name=""):
    # print all relevant metrics for classification
    print(f"\nModel: {model_name}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    if y_proba is not None:

# =============== STAGE 0: DATA LOADING & EXPLORATION ===============

def load_and_explore_data(filepath="creditcard.csv"):
    # load the kaggle credit card fraud dataset
    # the dataset has 28 anonymized PCA features (V1 to V28) + Amount + Class
    df = pd.read_csv(filepath)
    
    print_section("STAGE 0: DATA LOADING AND EXPLORATION")
    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    
    # check for missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\nTotal missing values: {total_missing}")
    
    # check class balance
    class_dist = df['Class'].value_counts()
    fraud_pct = class_dist.get(1, 0) / len(df) * 100
    print(f"\nClass distribution:")
    print(f"  Not Fraud (0): {class_dist.get(0, 0)} ({100-fraud_pct:.2f}%)")
    print(f"  Fraud (1): {class_dist.get(1, 0)} ({fraud_pct:.2f}%)")
    print(f"\nClass imbalance ratio: {class_dist.get(0, 0) / class_dist.get(1, 1):.1f}:1")
    
    # describe the data
    print(f"\nData Info:")
    print(df.info())
    
    return df
        print(f"ROC AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
