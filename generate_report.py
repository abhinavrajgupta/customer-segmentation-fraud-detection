from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.colors import darkblue, black
from reportlab.pdfgen import canvas

# Create the PDF
doc = SimpleDocTemplate("project_report.pdf", pagesize=letter)
story = []
styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=18,
    textColor=darkblue,
    alignment=TA_CENTER,
    spaceAfter=12
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=darkblue,
    spaceAfter=6
)

body_style = styles['Normal']

story.append(Paragraph("Customer Segmentation and Fraud Detection", title_style))
story.append(Paragraph("Project Summary Report", ParagraphStyle('SubTitle', parent=styles['Heading2'], fontSize=12, textColor=black, alignment=TA_CENTER, spaceAfter=20)))

story.append(Spacer(1, 10))

story.append(Paragraph("Overview", heading_style))
story.append(Paragraph("This project implements an end to end pipeline for credit card fraud detection and customer segmentation. It processes the Kaggle Credit Card Fraud Detection dataset through five sequential stages, from data loading through final model comparison.", body_style))
story.append(Spacer(1, 5))

story.append(Paragraph("Repository", heading_style))
story.append(Paragraph("Repository URL: https://github.com/abhinavrajgupta/customer-segmentation-fraud-detection", body_style))
story.append(Paragraph("Visibility: Public", body_style))
story.append(Paragraph("Primary Language: Python", body_style))
story.append(Spacer(1, 10))

story.append(Paragraph("Project Structure", heading_style))
story.append(Paragraph("data/ - Raw dataset and processed files", body_style))
story.append(Paragraph("src/ - Main solution code (solution.py)", body_style))
story.append(Paragraph("README.md - Project documentation and run instructions", body_style))
story.append(Paragraph("requirements.txt - Python dependencies", body_style))
story.append(Paragraph(".gitignore - Excludes data files, cache, and IDE configs", body_style))
story.append(Spacer(1, 10))

story.append(Paragraph("Pipeline Stages", heading_style))
story.append(Paragraph("Stage 0 - Data Loading and Exploration: Load the dataset, inspect columns, check for missing values, and analyze class distribution.", body_style))
story.append(Spacer(1, 4))
story.append(Paragraph("Stage 1 - Data Preprocessing: Create cyclical time features, apply log transform to Amount, standardize all features, and perform stratified train test split.", body_style))
story.append(Spacer(1, 4))
story.append(Paragraph("Stage 2 - Customer Segmentation: Apply PCA dimensionality reduction and use Gaussian Mixture Model clustering with 4 segments to identify customer groups.", body_style))
story.append(Spacer(1, 4))
story.append(Paragraph("Stage 3 - Fraud Detection: Train and evaluate four models - Isolation Forest (unsupervised), Random Forest (supervised), Logistic Regression, and Local Outlier Factor.", body_style))
story.append(Spacer(1, 4))
story.append(Paragraph("Stage 4 - Final Report and Summary: Compare model F1 scores side by side, analyze segment level transaction counts, and print pipeline completion message.", body_style))
story.append(Spacer(1, 10))

story.append(Paragraph("How to Run", heading_style))
story.append(Paragraph("1. Clone or open the repository", body_style))
story.append(Paragraph("2. Install dependencies: pip install -r requirements.txt", body_style))
story.append(Paragraph("3. Place creditcard.csv in the project directory", body_style))
story.append(Paragraph("4. Run: python src/solution.py", body_style))
story.append(Paragraph("5. View results printed to terminal for each stage", body_style))
story.append(Spacer(1, 10))

story.append(Paragraph("Key Libraries", heading_style))
story.append(Paragraph("numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, joblib, kaggle", body_style))
story.append(Spacer(1, 20))

story.append(Paragraph("Built by Abhinav Raj Gupta - May 2026", ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=black, alignment=TA_CENTER)))

# Build the PDF
doc.build(story)
print("PDF report generated: project_report.pdf")
