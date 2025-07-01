---
title: Stagergy 1
date: 2025-07-01
author: Your Name
cell_count: 12
score: 10
---

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

```


```python
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]

```


```python
df.columns = ['label', 'text']

```


```python
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = MultinomialNB()

```


```python
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

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
    
    


```python
sample = ["Congratulations! You won a $1000 prize. Click here to claim."]
sample_tfidf = vectorizer.transform(sample)
print("\nPrediction:", "Spam" if model.predict(sample_tfidf)[0] == 1 else "Ham")

```

    
    Prediction: Spam
    


```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

```


```python
categories = ['rec.sport.baseball', 'sci.space', 'comp.graphics', 'talk.politics.mideast']  
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

```


```python
model = LinearSVC()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=categories))

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
    
    


```python
sample = ["NASA successfully launched a new satellite into orbit last night."]
sample_tfidf = vectorizer.transform(sample)
predicted_category = model.predict(sample_tfidf)[0]
print("\nPredicted Category:", categories[predicted_category])
```

    
    Predicted Category: comp.graphics
    


```python
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import re
from bs4 import BeautifulSoup
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


```


```python
nltk.download('stopwords')
from nltk.corpus import stopwords


```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\stefi\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


---
**Score: 10**