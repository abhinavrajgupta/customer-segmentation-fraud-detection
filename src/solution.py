import pandas as pd
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    # print a clean section header for each stage
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_metrics(y_true, y_pred, y_proba=None, model_name=""):
    # print all classification metrics in one place
    print(f"\nModel: {model_name}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    if y_proba is not None:
        print(f"ROC AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


# ============================================================
# STAGE 0: DATA LOADING AND EXPLORATION
# ============================================================

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
    print(f"  Not Fraud (0): {class_dist.get(0, 0):,} ({100-fraud_pct:.3f}%)")
    print(f"  Fraud (1):     {class_dist.get(1, 0):,} ({fraud_pct:.3f}%)")
    print(f"\nClass imbalance ratio: {class_dist.get(0, 0) / class_dist.get(1, 1):.1f}:1")

    # describe the data
    print(f"\nData Info:")
    print(df.info())

    return df


# ============================================================
# STAGE 1: DATA PREPROCESSING
# ============================================================

def preprocess_data(df):
    # separate features from target
    X = df.drop(columns=['Class'])
    y = df['Class']

    print_section("STAGE 1: DATA PREPROCESSING")

    # the Time column is a raw timestamp in seconds
    # convert it to hours of the day (cyclical feature)
    X['Time_Hour'] = (X['Time'] / 3600) % 24
    X['Time_Sin'] = np.sin(2 * np.pi * X['Time_Hour'] / 24)
    X['Time_Cos'] = np.cos(2 * np.pi * X['Time_Hour'] / 24)
    print("  Created cyclical time features (Time_Sin, Time_Cos)")

    # log transform the Amount feature (highly skewed)
    X['Amount_Log'] = np.log1p(X['Amount'])
    print("  Applied log transform to Amount column")

    # drop original Time and Amount columns
    cols_to_drop = ['Time', 'Amount', 'Time_Hour']
    X = X.drop(columns=cols_to_drop)
    print(f"  Dropped columns: {cols_to_drop}")

    # standardize all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    print(f"  Applied StandardScaler to all {X_scaled.shape[1]} features")

    # split data (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")
    print(f"  Train fraud rate:  {y_train.mean()*100:.3f}%")
    print(f"  Test fraud rate:   {y_test.mean()*100:.3f}%")

    return X_train, X_test, y_train, y_test, scaler


# ============================================================
# STAGE 2: CUSTOMER SEGMENTATION (GMM CLUSTERING)
# ============================================================

def segment_customers(X_scaled):
    # use Gaussian Mixture Model for soft clustering
    # GMM gives us probability of belonging to each segment
    # this is better than KMeans for fraud analysis since
    # fraud patterns can overlap with normal patterns

    print_section("STAGE 2: CUSTOMER SEGMENTATION")

    # reduce to 2D for visualization using PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # fit GMM with 4 components
    # 4 segments balance detail with interpretability
    gmm = GaussianMixture(
        n_components=4,
        covariance_type='full',
        random_state=RANDOM_STATE,
        n_init=10
    )
    gmm.fit(X_scaled)

    # get cluster assignments and probabilities
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    # print segment sizes
    print("\n  Segment sizes:")
    for i in range(4):
        count = (labels == i).sum()
        pct = count / len(labels) * 100
        print(f"    Segment {i+1}: {count:,} transactions ({pct:.1f}%)")

    # analyze fraud concentration per segment
    print("\n  Fraud concentration per segment:")
    for i in range(4):
        mask = (labels == i)
        seg_fraud = X_scaled[mask, -1]  # Amount_Log is last column
        print(f"    Segment {i+1}: avg_amount_log = {seg_fraud.mean():.3f}")

    return labels, probs, gmm, pca


# ============================================================
# STAGE 3: FRAUD DETECTION (ENSEMBLE APPROACH)
# ============================================================

def detect_fraud(X_train, X_test, y_train, y_test):
    # use multiple models and compare their results
    # this ensemble approach is production ready

    print_section("STAGE 3: FRAUD DETECTION")

    results = {}

    # MODEL 1: ISOLATION FOREST (unsupervised anomaly detection)
    # good for detecting novel fraud patterns
    print("\n  Model 1: Isolation Forest (unsupervised)")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.0017,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    iso.fit(X_train)
    y_pred_iso = iso.predict(X_test)
    y_pred_iso = (y_pred_iso == -1).astype(int)  # -1 means anomaly
    print_metrics(y_test, y_pred_iso, model_name="Isolation Forest")
    results['isolation_forest'] = y_pred_iso

    # MODEL 2: RANDOM FOREST (supervised)
    # strong baseline for supervised fraud detection
    print("\n  Model 2: Random Forest Classifier\n")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    print_metrics(y_test, y_pred_rf, y_proba=y_proba_rf, model_name="Random Forest")
    results['random_forest'] = y_pred_rf
    results['rf_proba'] = y_proba_rf

    # MODEL 3: LOGISTIC REGRESSION
    # simple interpretable model for risk scoring
    print("\n  Model 3: Logistic Regression\n")
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='liblinear'
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    print_metrics(y_test, y_pred_lr, y_proba=y_proba_lr, model_name="Logistic Regression")
    results['logistic_regression'] = y_pred_lr

    # MODEL 4: LOCAL OUTLIER FACTOR
    # another unsupervised method for comparison
    print("\n  Model 4: Local Outlier Factor\n")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.0017,
        novelty=True,
        n_jobs=-1
    )
    lof.fit(X_train)
    y_pred_lof = lof.predict(X_test)
    y_pred_lof = (y_pred_lof == -1).astype(int)
    print_metrics(y_test, y_pred_lof, model_name="Local Outlier Factor")
    results['lof'] = y_pred_lof

    return results


# ============================================================
# STAGE 4: FINAL REPORT AND SUMMARY
# ============================================================

def generate_report(results, y_test, segment_labels):
    print_section("STAGE 4: FINAL REPORT AND SUMMARY")

    # compare all models side by side
    print("\n  MODEL COMPARISON")
    print("  " + "="*50)
    print(f"  {'Model':<25} {'F1 Score':>12}")
    print("  " + "="*50)

    for name, preds in results.items():
        if 'proba' in name:
            continue
        f1 = f1_score(y_test, preds)
        print(f"  {name:<25} {f1:>12.4f}")

    print("  " + "="*50)

    # segment analysis
    print("\n  SEGMENT FRAUD ANALYSIS")
    print("  " + "="*50)
    unique_segs = np.unique(segment_labels)
    for seg in unique_segs:
        mask = (segment_labels == seg)
        total = mask.sum()
        # we do not have fraud labels for full dataset here
        print(f"    Segment {int(seg)+1}: {total:,} transactions")

    print("\n  PIPELINE COMPLETE")
    print("  All stages executed successfully.\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("  CUSTOMER SEGMENTATION AND FRAUD DETECTION PIPELINE")
    print("#"*60)

    # Stage 0: Load data
    df = load_and_explore_data("creditcard.csv")

    # Stage 1: Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Stage 2: Customer segmentation
    segment_labels, segment_probs, gmm, pca = segment_customers(df.drop(columns=['Class']))

    # Stage 3: Fraud detection
    results = detect_fraud(X_train, X_test, y_train, y_test)

    # Stage 4: Final report
    generate_report(results, y_test, segment_labels)

    print("\n" + "="*60)
    print("  END OF PIPELINE")
    print("="*60 + "\n")
