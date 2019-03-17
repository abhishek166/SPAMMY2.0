import email
import email.parser
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

import re
import  nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#%%
#train dataset
mydir = os.getcwd()
files = os.listdir(mydir)
files =  files[1:]
frm = []
to1 = []
subj = []
body = []
for file in files:
    filepath = os.path.join(mydir, file)
    fp = open(filepath, encoding="latin1")
    msg = email.message_from_file(fp)
    payload = msg.get_payload()
    
    if type(payload) == type(list()):
        payload = payload[0]
    #print(payload)
    sub = msg.get('subject')
    to = msg.get('to')
    fr = msg.get('from')
    #print(sub)
    sub = str(sub)
    to = str(to)
    fr = str(fr)
    to1.append(to)
    subj.append(sub)
    frm.append(fr)
    payload = str(payload)
    body.append(payload)
    # To verify the code works
    #print(sub, "\n", payload[:2]);
    
#%%    
#Dataframe
dataset_train = pd.DataFrame()
dataset_train['body']=body
    
#%%
#Applying nltk(trimming body part of an email)
corpus_train = []
for i in range(len(dataset_train['body'])):
    review = re.sub('[^a-zA-Z]',' ', dataset_train['body'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_train.append(review)
    
#%%
#label file reading
df = pd.read_csv('Labels.csv')
y = df.iloc[:, 0].values    
    
#%%
#Create the bags of word model by applying filter
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus_train).toarray()

#%%
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#%%
#Applying Randomforest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, random_state=0)
classifier.fit(X_train, y_train )


y_pred = classifier.predict(X_test)

#%%
#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

def pretty_confusion_matrix(y_true, y_pred, labels = ['False', 'True']):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ['Predicted' + l for l in labels]
    df = pd.DataFrame(cm, index = labels, columns = pred_labels)
    return df
pretty_confusion_matrix(y_test, y_pred, ['Spam','Ham'])

#%%
plt.matshow(confusion_matrix(y_test, y_pred), cmap = plt.cm.binary, interpolation = 'nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('Expected label')

#%%
from sklearn.metrics import  precision_score, recall_score, f1_score
print("Precision:\t{:0.3f}".format(precision_score(y_test, y_pred)))
print("Recall:\t        {:0.3f}".format(recall_score(y_test, y_pred)))
print("F1 Score:\t{:0.3f}".format(f1_score(y_test, y_pred)))

#%%
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
