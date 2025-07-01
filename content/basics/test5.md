---
title: Test5
date: 2025-07-01
author: Your Name
cell_count: 2
score: 0
---

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

nltk.download('stopwords')
from nltk.corpus import stopwords

url = 'https://gitlab.com/rajacsp/datasets/raw/master/stack-overflow-data.csv'
response = requests.get(url)
df = pd.read_csv(BytesIO(response.content))

df = df[pd.notnull(df['tags'])]

REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").text
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

df['post'] = df['post'].apply(clean_text)

X = df['post']
y = df['tags']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

my_tags = [
    'java','html','asp.net','c#','ruby-on-rails','jquery','mysql','php',
    'ios','javascript','python','c','css','android','iphone','sql',
    'objective-c','c++','angularjs','.net'
]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=my_tags))
print("Accuracy:", accuracy_score(y_test, y_pred))

```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\stefi\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    

    Accuracy: 0.7395
    
    Classification Report:
    
                   precision    recall  f1-score   support
    
             java       0.63      0.65      0.64       613
             html       0.94      0.86      0.90       620
          asp.net       0.87      0.92      0.90       587
               c#       0.70      0.77      0.73       586
    ruby-on-rails       0.73      0.87      0.79       599
           jquery       0.72      0.51      0.60       589
            mysql       0.77      0.74      0.75       594
              php       0.69      0.89      0.78       610
              ios       0.63      0.59      0.61       617
       javascript       0.57      0.65      0.60       587
           python       0.70      0.50      0.59       611
                c       0.79      0.79      0.79       594
              css       0.84      0.59      0.69       619
          android       0.66      0.84      0.74       574
           iphone       0.64      0.83      0.72       584
              sql       0.66      0.64      0.65       578
      objective-c       0.79      0.77      0.78       591
              c++       0.89      0.83      0.86       608
        angularjs       0.94      0.89      0.91       638
             .net       0.74      0.66      0.70       601
    
         accuracy                           0.74     12000
        macro avg       0.74      0.74      0.74     12000
     weighted avg       0.75      0.74      0.74     12000
    
    Accuracy: 0.7395
    


```python

```


---
**Score: 0**