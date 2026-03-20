import pandas as pd
import numpy as np
import re


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.svm import LinearSVC


columns=['target','id','date','flag','user','text']
df=pd.read_csv("sentiment140.csv",encoding='latin-1',names=columns)
df=df[['target','text']]
df['target']=df['target'].replace(4,1)
print(df.head())
print(df['target'].value_counts())

def clean_text(text):
    text=text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) #remove URLs
    text=re.sub(r'@\w+','',text) # remove mentions 
    text=re.sub(r'[^a-zA-Z]',' ',text) # remove special characters
    return text
    
    
df['clean_text']=df['text'].apply(clean_text)
print(df[['text', 'clean_text']].head())

vectorizer=TfidfVectorizer(max_features=6000, ngram_range=(1,2))
X=vectorizer.fit_transform(df['clean_text'])
y=df['target'] 
#Logistic Regression Implementation
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
print("\n Classification Reprot:\n")
print(classification_report(y_test,y_pred))
print("\n Confusion Matrix: \n")
print(confusion_matrix(y_test,y_pred))

# Testing on some input data
def predict_sentiment(text):
    text=clean_text(text)
    text=vectorizer.transform([text])
    prediction=model.predict(text)[0]
    return "Positive" if prediction==1 else "Negative"
print(predict_sentiment("I love this phone"))
print(predict_sentiment("This is the worst experience ever"))

#Bernoulli Naives Bayes Implementation
bnb_model=BernoulliNB()
bnb_model.fit(X_train,y_train)
y_pred_bnb=bnb_model.predict(X_test)
print("Bernoulli NB Accuracy:", accuracy_score(y_test, y_pred_bnb))
print("\n Classification Report:\n")
print(classification_report(y_test,y_pred_bnb))
print("\n Confusion Matrix: \n")
print(confusion_matrix(y_test,y_pred_bnb))
# Testing on some input data
def predict_sentiment_bnb(text):
    text=clean_text(text)
    text=vectorizer.transform([text])
    prediction=bnb_model.predict(text)[0]
    return "Positive" if prediction ==1 else "Negative"
print(predict_sentiment_bnb("I love this phone"))
print(predict_sentiment_bnb("This is worst experience ever"))

# Support Vector Machine Implementation
svm_model=LinearSVC()
svm_model.fit(X_train, y_train)
y_pred_svm= svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_svm))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_svm))

