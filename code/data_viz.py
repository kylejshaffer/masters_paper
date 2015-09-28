import os, cPickle
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

os.chdir('/Users/kylefth/Desktop/SILS_Courses/MS_Paper/data')

with open('analysis_forums.pickle', 'r') as infile:
    data = cPickle.load(infile)
np_labels = data['labels']
labs = []
for i in np_labels:
    for j in i:
        labs.append(j)
'''
# Counts of distribution of labels (bar)
# Integer labels mapped from strings
string_labs = ['Q', 'A', 'I', 'R', 'P', 'N', 'O']
ints_of_labs = [0,1,2,3,4,5,6]
counts = [labs.count(i) for i in set(labs)]
plt.bar(ints_of_labs, counts, align='center')
plt.xticks(ints_of_labs, string_labs)
plt.xlabel('Speech Acts')
plt.ylabel('Frequency')
plt.title('Distribution of Speech Acts in Dataset')
plt.show()

# Counts of distribution of threads (hist)
thread_lengths = [len(i) for i in np_labels]
#plt.hist(thread_lengths, bins=len(thread_lengths), color='steelblue')
#plt.hist(thread_lengths, bins=50, color='steelblue')
#plt.plot(thread_lengths, color='steelblue')
hist_data = np.histogram(thread_lengths, bins = len(thread_lengths))[0]
sns.distplot(thread_lengths, hist=False, color=sns.color_palette("muted")[0], kde_kws={'shade': True})
#plt.plot(thread_lengths, hist_data, color='steelblue', alpha=0.5)
plt.xlim(1, max(thread_lengths))
plt.ylabel('Proportion of Threads')
plt.xlabel('Number of Posts in Thread')
plt.title('Distribution of Thread Lengths')
plt.show()

# Bar charts of model performance
# Precision
lr_p = np.array([.238, .421, .264, .083, .464, .054, .361])
crf_p = np.array([.45, .429, .431, .203, .46, .05, .446])
rand_p = np.array([.124, .27, .09, .028, .339, .02, .131])

# Setting up data frames for precision reproting
precision_df = pd.DataFrame(rand_p, columns = ["BASE"])
precision_df['LR'] = lr_p
precision_df['CRF'] = crf_p
precision_df.index = ['Q', 'A', 'I', 'R', 'P', 'N', 'O']

# Plotting
precision_df.plot(kind='bar')
plt.ylabel('Precision')
plt.show()

# F1 Scores
lr_f1 = np.array([.208, .36, .274, .122, .424, .087, .396])
crf_f1 = np.array([.397, .446, .327, .161, .521, .05, .373])
rand_f1 = np.array([.22, .425, .165, .055, .506, .039, .032])

# Setting up data frames for precision reproting
f1_df = pd.DataFrame(rand_f1, columns = ["BASE"])
f1_df['LR'] = lr_f1
f1_df['CRF'] = crf_f1
f1_df.index = ['Q', 'A', 'I', 'R', 'P', 'N', 'O']

# Plotting
f1_df.plot(kind='bar')
plt.ylabel('F1 Score')
plt.show()

# Confusion matrix for classification results

# Want to show this for linear model and structured model

import matplotlib.pyplot as plt
plt.style.use('ggplot')
confusion = confusion_matrix(pred, y_true)
plt.matshow(confusion)
plt.colorbar()
plt.title()
plt.show()
'''

# Matrix showing transitions learned by CRF
# After ssvm has been fit on partiular fold
'''
speech_acts = 'QAIRPNO'
# Probably want to choose this randomly from the 10-fold CV
ssvm.fit(X_train, y_train)
plt.title('Transition Parameters of CRF Model')
plt.xticks(np.arange(7),speech_acts)
plt.yticks(np.arange(7), speech_acts)
plt.matshow(ssvm.w[-49:].reshape(7,7))
'''