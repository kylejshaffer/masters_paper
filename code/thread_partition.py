''' Following OCR example for data structure
    for letters['data'] => each element of data is a thread
    for letters['labels'] => each element of labels is the list of SAs corresponding to the posts within that thread
    for letters['folds'] => each element of folds is an ID for the fold for k-fold CV
'''

import os, json, cPickle
import pandas as pd
import numpy as np
from operator import itemgetter
# Paths for re-constructing threads and features
thread_dir = '/Users/kylefth/Desktop/SILS_Courses/MS_Paper/data/threads' 
data_dir = '/Users/kylefth/Desktop/SILS_Courses/MS_Paper/data/features'
label_dir = '/Users/kylefth/Desktop/SILS_Courses/MS_Paper/data'
os.chdir(data_dir)

# Iteratively combining feature-sets together into one data-frame
question_unigrams = ['anyon', 'doe', 'what', 'how', 'anybodi', 'ani', 'can', 'question', 'why', 'would', 'take', 'recip', 'note', 'where', 'wonder', 'email', 'know', 'thought', 'regard', 'ontolog', 'id']
question_unigrams = [i.upper() for i in question_unigrams]
answer_unigrams = ['thank', 'exampl', 'href', 'depend', 'describ', 'http', 'target', 'object', 'meta', 'tag', 'scheme', 'org', 'content', 'mean', 'element', 'defin', 'not', 'format', 'that', 'vocabulari', 'id']
answer_unigrams = [i.upper() for i in answer_unigrams]
issue_unigrams = ['strong', 'typo', 'factual', 'browser', 'omit', 'error', 'problem', 'detail', 'addit', 'window', 'chrome', 'mistak', 'firefox', 'homework', 'issu', 'quiz', 'messag', 'type', 'question', 'submiss', 'id']
issue_unigrams = [i.upper() for i in issue_unigrams]
issueres_unigrams = ['apolog', 'resolv', 'fix', 'caus', 'sorri', 'now', 'remov', '001', 'updat', 'homework', 'been', 'encount', 'score', 'appar', 'issu', 'sincer', 'instead', 'credit', 'player', 'submit', 'id']
issueres_unigrams = [i.upper() for i in issueres_unigrams]
posack_unigrams = ['thank', 'cathi', 'agre', 'http', 'href', 'great', 'target', 'org', 'can', 'not', 'pomerantz', 'which', 'page', 'question', 'much', 'the', 'problem', 'name', 'titl', 'amaz', 'id']
posack_unigrams = [i.upper() for i in posack_unigrams]
negack_unigrams = ['disagre', 'frustrat', 'anonym', 'less', 'code', 'necessarili', 'exercis', 'date', 'white', 'hate', 'not', 'turn', 'consider', 'disappoint', 'accept', 'get', 'challeng', 'simpli', 'express', 'perhap', 'id']
negack_unigrams = [i.upper() for i in negack_unigrams]
other_unigrams = ['que', 'para', 'por', 'gracia', 'con','pero', 'todo', 'curso', 'del', 'the', 'como', 'mucha', 'est', 'reput', 'thank', 'that', 'com', 'you', 'grei', 'stat', 'id']
other_unigrams = [i.upper() for i in other_unigrams]

feature_files = os.listdir(os.getcwd())
unigram_features = [i for i in feature_files if 'unigram' in i]
rest_features = [i for i in feature_files if not('unigram' in i)]

def unigram_dataframes(unigram, filename):
    df = pd.read_csv(filename, sep='\t')
    if unigram == 'question': unigram_list = question_unigrams
    if unigram == 'answer' :unigram_list = answer_unigrams
    if unigram == 'issue': unigram_list = issue_unigrams
    if unigram == 'issue_resolution': unigram_list = issueres_unigrams
    if unigram == 'positive_ack': unigram_list = posack_unigrams
    if unigram == 'negative_ack': unigram_list = negack_unigrams
    else: unigram_list = other_unigrams 
    cols_to_drop = [col for col in df.columns if not(col in unigram_list)]
    df = df.drop(cols_to_drop, axis=1)
    return df
    
def feature_dataframes(filename):
    df = pd.read_csv(filename, sep='\t')
    cols_to_drop = [i for i in df.columns if 'Unnamed' in i]
    df = df.drop(cols_to_drop, axis=1)
    return df
    
features = ''
for f in rest_features:
    data = feature_dataframes(f)
    if type(features) == str:
        features = data
    else:
        features = features.merge(data, on='ID')

u = ''
for i in unigram_features:
    unigram = i[len('unigram')+1:-4]
    data = unigram_dataframes(unigram, i)
    if type(u) == str:
        u = data
    else: 
        u = u.merge(data, on='ID')

print u.shape
print features.shape

# Get rid of Unnamed feature
delete_list = [i for i in features.columns if 'Unnamed' in i]
for col in delete_list:
    del features[col]
# Save all features data frame to CSV
features.to_csv('all_features.csv')
# features = pd.read_csv('all_features.csv')
# Matching feature-set ID to 
features_json = features.to_json(orient='records')
features_json = json.loads(features_json)
for f in features_json:
    f['ID'] = int(f['ID'])

# Loading in labels and calculating majority vote
os.chdir(label_dir)
labels = pd.read_csv('labels.tsv', sep='\t')
labels_dict = labels.to_json(orient='records')
labels_dict = json.loads(labels_dict)
for lab in labels_dict:
    ID = lab['ID']
    del lab['ID']
    lab['LABEL'] = max(lab.iteritems(), key=itemgetter(1))[0]
    lab['ID'] = ID
labels = pd.DataFrame(labels_dict)
labels = labels[['ID', 'LABEL']]
# Converting labels to integer mapping
# Try starting your labels at 0 thru n-1
labels.ix[labels.LABEL=='Q', 'INT_LABEL'] = 0 # Question
labels.ix[labels.LABEL=='A', 'INT_LABEL'] = 1 # Answer
labels.ix[labels.LABEL=='I', 'INT_LABEL'] = 2 # Issue
labels.ix[labels.LABEL=='R', 'INT_LABEL'] = 3 # Issue Resolution
labels.ix[labels.LABEL=='P', 'INT_LABEL'] = 4 # Positive Acknolwedgement
labels.ix[labels.LABEL=='N', 'INT_LABEL'] = 5 # Negative Acknolwedgement
labels.ix[labels.LABEL=='O', 'INT_LABEL'] = 6 # Other
print labels.head(20)
labels = json.loads(labels.to_json(orient='records'))
for lab in labels:
    lab['INT_LABEL'] = int(lab['INT_LABEL'])

data = []
grouped_labels = []
os.chdir(thread_dir)
threads = [i for i in os.listdir(os.getcwd()) if i[-4:] == '.txt']
for num, thread in enumerate(threads):
    instance = []
    thread_labels = []
    with open(thread, 'r') as infile:
        posts = infile.readlines()
        posts = [int(i.strip()) for i in posts]
        for feature in features_json:
            feature['THREAD_ID'] = num
            if feature['ID'] in posts:
                instance.append(feature)
        for lab in labels:
            lab['THREAD_ID'] = num
            if lab['ID'] in posts:
                thread_labels.append(lab)
    # instance_df = np.array(pd.DataFrame(instance))
    grouped_labels.append(thread_labels)
    data.append(instance)
# Sanity check looping through data and labels to make sure lengths line up
for i,j in enumerate(data):
    if len(j) != len(grouped_labels[i]): print i; break
for d in data:
    for e in d:
        del e['ID']
        del e['THREAD_ID']

# Re-formatting labels as NP arrays
np_labels = []
for lab in grouped_labels:
    thread_labs = np.array([i['INT_LABEL'] for i in lab])
    np_labels.append(thread_labs)

np_data = []
for thread in data:
    np_thread = np.array(pd.DataFrame(thread))
    np_data.append(np_thread)
# Need to construct folds with some kind of round-robin labeling
folds = range(10) * 42
np_folds = np.array(folds, dtype='uint8')

# Initializing dict to store data structure with everything
# converted to NP arrays
np_forums = dict()
np_forums['data'] = np_data
np_forums['labels'] = np_labels
np_forums['folds'] = np_folds
os.chdir(label_dir)
with open('analysis_forums.pickle', 'w') as outfile:
    cPickle.dump(np_forums, outfile)
