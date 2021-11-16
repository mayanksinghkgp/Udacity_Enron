# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:27:38 2019

@author: Mayank 
"""
import sys
import pickle
import numpy as np
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock',
       'shared_receipt_with_poi', 'expenses', 'from_messages', 'other','long_term_incentive', 'loan_advances', 'total_stock_value'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Task 2: Remove outliers


data_dict.pop('TOTAL',0)

ind = []
for i in data_dict:
    nan_count=0
    for j in data_dict[i]:
        if data_dict[i][j] == 'NaN':
            nan_count+=1
    if nan_count>=20:
        ind.append(i)
#print "index:", ind
for i in ind:
    data_dict.pop(i,0)

### Task 3: Create new feature(s)

for i in data_dict:
    if data_dict[i]['from_this_person_to_poi'] != 'NaN' and data_dict[i]['from_messages'] != 'NaN':
        data_dict[i]['ratio_1'] = data_dict[i]['from_this_person_to_poi']/float(data_dict[i]['from_messages'])
    else:
        data_dict[i]['ratio_1'] = 0
    if data_dict[i]['from_poi_to_this_person'] != 'NaN' and data_dict[i]['to_messages'] != 'NaN':
        data_dict[i]['ratio_2'] = data_dict[i]['from_poi_to_this_person']/float(data_dict[i]['to_messages'])
    else:
        data_dict[i]['ratio_2'] = 0

features_list.append('ratio_1')
features_list.append('ratio_2')

#print features_list

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#test_classifier(clf,data_dict,features_list)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(max_depth=2, random_state=42)
#t0 = time()
kbest = SelectKBest(f_classif)
pipeline1 = Pipeline([('kbest', kbest), ('classifier', clf1)])
grid_search = GridSearchCV(pipeline1, {'kbest__k': [8,9,10,11,12,13], 'classifier__n_estimators': [50,100]})
grid_search.fit(features, labels)
#print grid_search.best_score_
#print grid_search.best_params_


rem_feat = grid_search.best_params_['kbest__k']
n_est = grid_search.best_params_['classifier__n_estimators']
print 'Selecting best ', rem_feat , ' of all features'

#print "training time:", round(time()-t0, 3), "s"
clf2 = RandomForestClassifier(n_estimators=n_est,max_depth=2, random_state=42)
clf2.fit(features, labels)
importances = clf2.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf1.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for i in range(0,len(features_list)-rem_feat):
    features_list.remove(features_list[indices[rem_feat]+i+1])
#features_list.remove(features_list[indices[-1]+1])    
print 'Selected Features: ', features_list

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)
X = np.array(features)
y = np.array(labels)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

parameters = {
    'base_estimator__criterion': ('gini', 'entropy'),
    'base_estimator__min_samples_leaf':range(1, 50, 5),
    'base_estimator__max_depth': range(1, 10),
    'n_estimators': range(1,10),
    'algorithm':("SAMME", "SAMME.R")
}

clf = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(random_state=42), random_state=42), 
                      parameters, make_scorer(f1_score))
clf= clf.fit(X_train, y_train)
#print clf.best_score_
#print clf.best_params_
clf = clf.best_estimator_

estimators = [('scaler', MinMaxScaler()),
                        ('clf', clf)]
clf = Pipeline(estimators)


#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4, random_state=42),algorithm="SAMME",n_estimators=10, random_state=42)
#test_classifier(clf, my_dataset, features_list)

# Example starting point. Try investigating other evaluation techniques!
#test_classifier(clf, my_dataset, features_list)

test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


