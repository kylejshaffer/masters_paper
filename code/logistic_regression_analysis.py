import os, cPickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import islice
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Function we'll need later for feature scaling
def pair(seq, n):
    z = (islice(seq, i, None) for i in range(n))
    return zip(*z)

# Data setup
os.chdir(data_dir)
with open('analysis_forums.pickle', 'r') as infile:
    forums = cPickle.load(infile)

X, y, folds = forums['data'], forums['labels'], forums['folds']
X, y = np.array(X), np.array(y)

maxent = LogisticRegression(C=1., class_weight='auto')

transition_states = ''
total_confusion_matrix = ''
# Starter code for 10-fold CV for CRF
for fold in xrange(len(set(folds))):
    # Training set
    X_train, y_train = X[folds != fold], y[folds != fold]
    # Test set
    X_test, y_test = X[folds == fold], y[folds == fold]
    
    # Performing feature scaling
    Xtrain_scaler = np.vstack(X_train)
    scaler = MinMaxScaler().fit(Xtrain_scaler)
    Xtrain_scaler = scaler.transform(Xtrain_scaler)
    # Flattening out training labels
    y_train = np.hstack(y_train)
    
    # Same procedure for test set
    Xtest_scaler = np.vstack(X_test)
    Xtest_scaler = scaler.transform(Xtest_scaler)
    # Flattening out test labels
    y_test = np.hstack(y_test)
    
    maxent.fit(Xtrain_scaler, y_train)
    
    predicted = maxent.predict(Xtest_scaler)
    
    # Metrics
    fold_cm = confusion_matrix(y_test, predicted)
    if type(total_confusion_matrix) == str:
        total_confusion_matrix = fold_cm
    else: total_confusion_matrix = total_confusion_matrix + fold_cm
    
    report = classification_report(y_test, predicted)
    for line in report.split('\n'):
        print line
        
speech_acts = 'QAIRPNO'
normalized_crf_confusion_matrix = total_confusion_matrix / 10.
plt.matshow(normalized_crf_confusion_matrix)
plt.yticks(np.arange(7), speech_acts)
plt.xticks(np.arange(7), speech_acts)
plt.colorbar()
plt.title('Confusion Matrix for Logistic Regression Model')
plt.show() 
