🛒 Q-Commerce Sentiment Analysis

Sentiment Prediction Using Machine Learning for Q-Commerce Consumer Opinions

📌 Project Overview

This project performs sentiment analysis (Positive / Neutral / Negative) on Q-Commerce consumer survey data using Natural Language Processing (NLP) and Machine Learning.

Since the given dataset is survey-based and does not contain sentiment labels, sentiment is derived from opinion-based responses, and a machine learning model is trained on the derived labels.

🧠 Technologies Used

Python 3.8+

Pandas

NLTK

Scikit-learn

TF-IDF Vectorizer

Logistic Regression

Joblib

📂 Project Structure
QCommerce_Sentiment_Analysis_Final/
│
├── data/
│   └── raw/
│       └── Quick_Commerce_Consumer_Behavior.xlsx
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── labeling.py
│   ├── vectorization.py
│   └── model_training.py
│
├── models/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── test_prediction.py
├── main.py
├── requirements.txt
└── README.md

⚙️ Step-by-Step Setup & Execution Guide
✅ STEP 1: Install Python

Ensure Python 3.8 or above is installed.

Check version:

python --version

✅ STEP 2: Extract Project

Unzip the project folder:

QCommerce_Sentiment_Analysis_Final

✅ STEP 3: Place Dataset (VERY IMPORTANT)

Copy the given Excel file to this exact path:

data/raw/Quick_Commerce_Consumer_Behavior.xlsx


⚠️ File name and location must match exactly.

✅ STEP 4: Open Terminal / Command Prompt

Navigate to the project root folder.

Windows

cd path\to\QCommerce_Sentiment_Analysis_Final


Linux / macOS

cd path/to/QCommerce_Sentiment_Analysis_Final

✅ STEP 5: Create Virtual Environment (Recommended)
python -m venv venv

Activate Virtual Environment

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

✅ STEP 6: Install Required Packages
pip install -r requirements.txt


If NLTK stopwords error occurs:

python -c "import nltk; nltk.download('stopwords')"

✅ STEP 7: Train the Machine Learning Model

Run the main training script:

python main.py

Expected Output
Model trained successfully


This creates:

models/
├── sentiment_model.pkl
└── vectorizer.pkl

✅ STEP 8: Run Live Sentiment Prediction (Demo)

Run the prediction script:

python test_prediction.py

Example Demo
Enter customer opinion: I am satisfied with fast delivery
Predicted Sentiment: Positive

Enter customer opinion: Concern about late delivery
Predicted Sentiment: Negative

Enter customer opinion: More product variety expected
Predicted Sentiment: Neutral


Type exit to stop.