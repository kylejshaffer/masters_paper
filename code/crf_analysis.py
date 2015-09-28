import os, cPickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import islice
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM
from sklearn.preprocessing import StandardScaler
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
print 'Train - test pairs constructed'

# Model setup
crf = ChainCRF()
ssvm = OneSlackSSVM(model=crf, C=1.)
print 'Models loaded'
#X_train, y_train = X[folds != 0], y[folds != 0]
#X_test, y_test = X[folds == 0], y[folds == 0]

#ssvm.fit(X_train, y_train)
# predicted = ssvm.predict(X_test)


transition_states = ''
total_confusion_matrix = ''

# 10-fold CV for CRF
for fold in range(len(set(folds))):
    # Training set
    X_train, y_train = X[folds != fold], y[folds != fold]
    # Test set
    X_test, y_test = X[folds == fold], y[folds == fold]
    print 'data loaded'
    # Performing feature scaling
    Xtrain_scaler = np.vstack(X_train)
    scaler = StandardScaler().fit(Xtrain_scaler)
    Xtrain_scaler = scaler.transform(Xtrain_scaler)

    # Inserting scaled training set back into data structure
    index_list = [i.shape[0] for i in X_train]
    new_index_list = []
    running_sum = 0
    for index in index_list:
        running_sum += index
        new_index_list.append(running_sum)
    new_index_list = pair(new_index_list, 2)
    new_index_list.insert(0, (0, new_index_list[0][0]))

    Xtrain_scale = []
    for num in new_index_list:
        new_array = Xtrain_scaler[num[0]:num[1],:]
        Xtrain_scale.append(new_array)
    Xtrain_scale = np.array(Xtrain_scale)
    print Xtrain_scale.shape
    print 'Training set scaled'
    # Same procedure for test set
    Xtest_scaler = np.vstack(X_test)
    Xtest_scaler = scaler.transform(Xtest_scaler)
    index_list = [i.shape[0] for i in X_test]
    new_index_list = []
    running_sum = 0

    for index in index_list:
        running_sum += index
        new_index_list.append(running_sum)
    new_index_list = pair(new_index_list, 2)
    new_index_list.insert(0, (0, new_index_list[0][0]))

    Xtest_scale = []
    for num in new_index_list:
        new_array = Xtest_scaler[num[0]:num[1],:]
        Xtest_scale.append(new_array)
    Xtest_scale = np.array(Xtest_scale)
    print Xtest_scale.shape
    print 'Test set scaled'
    ssvm.fit(Xtrain_scale, y_train)
    print 'Model fit'

    # Storing model weights for plot of transition probabilities
    if type(transition_states) == str:
        transition_states = ssvm.w[-49:].reshape(7,7)
    else: transition_states = transition_states + ssvm.w[-49:].reshape(7,7)
    
    predicted = ssvm.predict(Xtest_scale)
    print 'Predictions made'
    # Metrics
    fold_cm = confusion_matrix(np.hstack(y_test), np.hstack(predicted))
    if type(total_confusion_matrix) == str:
        total_confusion_matrix = fold_cm
    else: total_confusion_matrix = total_confusion_matrix + fold_cm
    
    report = classification_report(np.hstack(y_test), np.hstack(predicted))
    for line in report.split('\n'):
        print line
    print 'Metrics written'

# Code for plotting matrix depicting transition probabilities
# for the CRF model averaged across 10 folds of CV.
normalized_states = transition_states / 10.
speech_acts = 'QAIRPNO'
plt.matshow(normalized_states)
plt.xticks(np.arange(7), speech_acts)
plt.yticks(np.arange(7), speech_acts)
plt.colorbar()
plt.title('Transition Parameters of CRF Model')
plt.show()

speech_acts = 'QAIRPNO'
normalized_crf_confusion_matrix = total_confusion_matrix / 10.
plt.matshow(normalized_crf_confusion_matrix)
plt.yticks(np.arange(7), speech_acts)
plt.xticks(np.arange(7), speech_acts)
plt.colorbar()
plt.title('Confusion Matrix for CRF Model')
plt.show() 
