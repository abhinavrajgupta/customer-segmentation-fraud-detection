# Customer Segmentation and Fraud Detection

End to end pipeline for credit card fraud detection and customer segmentation

## Project Structure

data/           raw dataset and processed files
src/            main solution code
  solution.py   complete 5 stage pipeline
README.md       this file
requirements.txt

## Stages

Stage 0: Data Loading and Exploration
  Load dataset, check class balance, inspect columns

Stage 1: Data Preprocessing
  Handle class imbalance, create time features, scale features

Stage 2: Customer Segmentation
  GMM clustering to identify customer groups

Stage 3: Fraud Detection
  Multiple models: Isolation Forest, Random Forest, Logistic Regression, Local Outlier Factor

Stage 4: Final Report and Summary
  Model comparison and segment level fraud statistics

## How to Run

1. Clone or open this repo
2. Install dependencies:
   pip install -r requirements.txt
3. Make sure creditcard.csv is in the src folder
4. Run:
   python src/solution.py
5. Results print to terminal for each stage

## Requirements

numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
kaggle
joblib

## Dataset

Kaggle Credit Card Fraud Detection dataset
