---
title: Test3
date: 2025-07-01
author: Your Name
cell_count: 2
score: 0
---

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
categories = ['rec.sport.baseball', 'sci.space', 'comp.graphics', 'talk.politics.mideast']  
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LinearSVC()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=categories))
sample = ["NASA successfully launched a new satellite into orbit last night."]
sample_tfidf = vectorizer.transform(sample)
predicted_category = model.predict(sample_tfidf)[0]
print("\nPredicted Category:", categories[predicted_category])
```

    Confusion Matrix:
     [[199  13   8   0]
     [  3 169   3   3]
     [  6  12 165   6]
     [  3   9   1 179]]
    
    Classification Report:
                            precision    recall  f1-score   support
    
       rec.sport.baseball       0.94      0.90      0.92       220
                sci.space       0.83      0.95      0.89       178
            comp.graphics       0.93      0.87      0.90       189
    talk.politics.mideast       0.95      0.93      0.94       192
    
                 accuracy                           0.91       779
                macro avg       0.91      0.91      0.91       779
             weighted avg       0.92      0.91      0.91       779
    
    
    Predicted Category: comp.graphics
    


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('imdb_reviews.csv')
df = df[['review', 'sentiment']]
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sample = ["This movie was a masterpiece. I loved every minute of it!"]
sample_tfidf = vectorizer.transform(sample)
print("\nPrediction:", "Positive" if model.predict(sample_tfidf)[0] == 1 else "Negative")

```

    Confusion Matrix:
     [[4351  610]
     [ 448 4591]]
    
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.91      0.88      0.89      4961
               1       0.88      0.91      0.90      5039
    
        accuracy                           0.89     10000
       macro avg       0.89      0.89      0.89     10000
    weighted avg       0.89      0.89      0.89     10000
    
    
    Prediction: Positive
    


---
**Score: 0**