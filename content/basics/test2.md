---
title: Test2
date: 2025-07-01
author: Your Name
cell_count: 2
score: 0
---

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sample = ["Congratulations! You won a $1000 prize. Click here to claim."]
sample_tfidf = vectorizer.transform(sample)
print("\nPrediction:", "Spam" if model.predict(sample_tfidf)[0] == 1 else "Ham")

```

    Confusion Matrix:
     [[965   0]
     [ 37 113]]
    
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.96      1.00      0.98       965
               1       1.00      0.75      0.86       150
    
        accuracy                           0.97      1115
       macro avg       0.98      0.88      0.92      1115
    weighted avg       0.97      0.97      0.96      1115
    
    
    Prediction: Spam
    


```python

```


---
**Score: 0**