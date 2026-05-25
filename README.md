# Customer Segmentation and Fraud Detection

## Overview

This project builds a complete pipeline for analyzing credit card transactions. It groups customers into meaningful segments and flags suspicious transactions. The code follows a clean stage by stage flow so each part can be tested and understood on its own.

## Project Structure
```
repo_root/
  src/
    solution.py       main pipeline code
  requirements.txt    python dependencies
  README.md           this file
```

## Pipeline Stages

The solution is split into five stages. Each stage is a separate section in the code. You can read or run them one at a time.

* Stage 0: Setup and data loading
* Stage 1: Data cleaning and preprocessing
* Stage 2: Customer segmentation using clustering
* Stage 3: Fraud detection with Isolation Forest
* Stage 4: Final report and model summary

## How to Run

1. Open this repo in GitHub Codespace or clone it locally
2. Install the dependencies listed in requirements.txt
3. Place the creditcard.csv file in the repo root folder
4. Run the following command:

```bash
python src/solution.py
```
5. The script will print results for each stage in the terminal

## Requirements

Install packages from requirements.txt:

```bash
pip install -r requirements.txt
```

## What You Get

After running the script you will see:

* Basic data info and null counts
* A summary of preprocessing steps applied
* Customer segments with cluster centers
* Fraud flagging results with sample transactions
* A final summary with key numbers
