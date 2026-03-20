📊 Twitter Sentiment Analysis

This project performs sentiment analysis on tweets to classify them as positive or negative using Natural Language Processing (NLP) and machine learning techniques.

🚀 Overview

The project uses the Sentiment140 dataset to train multiple machine learning models and evaluate their performance on tweet sentiment classification. It includes data preprocessing, feature extraction, model training, and prediction.

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Regular Expressions (re)

📂 Dataset

Dataset Used: Sentiment140

Contains tweets labeled as:

0 → Negative

4 → Positive (converted to 1 in this project)

🔄 Workflow

Data Loading & Cleaning

Removed URLs, mentions, and special characters

Converted text to lowercase

Feature Extraction

Used TF-IDF Vectorization

Included unigrams and bigrams

Model Training

Logistic Regression

Bernoulli Naive Bayes

Support Vector Machine (LinearSVC)

Evaluation

Accuracy Score

Classification Report

Confusion Matrix

Prediction

Custom input prediction for sentiment classification

📈 Models Performance

The following models were implemented and compared:

Logistic Regression

Bernoulli Naive Bayes

Support Vector Machine (SVM)

Each model was evaluated using standard metrics to compare performance.

💡 Example Predictions
predict_sentiment("I love this phone")  
# Output: Positive

predict_sentiment("This is the worst experience ever")  
# Output: Negative
📚 Reference

This project was implemented with guidance from:

https://www.geeksforgeeks.org/python/twitter-sentiment-analysis-using-python/

📌 Key Learnings

Text preprocessing techniques in NLP

Feature extraction using TF-IDF

Comparing multiple ML models for classification

Model evaluation using standard metrics

🔮 Future Improvements

Add deep learning models (LSTM, BERT)

Improve preprocessing (stopwords, stemming, lemmatization)

Deploy as a web application

Use real-time tweet data via API
