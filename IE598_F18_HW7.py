#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:22:07 2018

@author: kirktsui
"""

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd



df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 
                    'Malic acid', 'Ash', 
                    'Alcalinity of ash', 
                    'Magnesium', 'Total phenols', 
                    'Flavanoids', 'Nonflavanoid phenols',
                    'Proanthocyanins', 
                    'Color intensity', 'Hue', 
                    'OD280/OD315 of diluted wines', 
                    'Proline']


y = df_wine['Class label']
X = df_wine[['Alcohol', 
                    'Malic acid', 'Ash', 
                    'Alcalinity of ash', 
                    'Magnesium', 'Total phenols', 
                    'Flavanoids', 'Nonflavanoid phenols',
                    'Proanthocyanins', 
                    'Color intensity', 'Hue', 
                    'OD280/OD315 of diluted wines', 
                    'Proline']]
X.columns = ['Alcohol', 
                    'Malic acid', 'Ash', 
                    'Alcalinity of ash', 
                    'Magnesium', 'Total phenols', 
                    'Flavanoids', 'Nonflavanoid phenols',
                    'Proanthocyanins', 
                    'Color intensity', 'Hue', 
                    'OD280/OD315 of diluted wines', 
                    'Proline']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3, stratify = y)
             


######Cross Validation#################
score_means = []
score_stds = []

for i in [5,10,20,50,100,200]:
    forest = RandomForestClassifier(criterion = 'gini', n_estimators = i, random_state = 5, n_jobs = -1)
    cv_score = cross_val_score (estimator = forest, X=X_train, y = y_train, cv =10, n_jobs = -1)
    score_mean = np.mean(cv_score)
    score_means.append(score_mean)
    score_std = np.std(cv_score)
    score_stds.append(score_std)
    print('When N_estimator = %.0f, accuracy = %.4f +- %.4f'%(i, score_mean,score_std))
    
optimal_forest = RandomForestClassifier(criterion = 'gini', n_estimators = 50, random_state = 5, n_jobs = -1)
optimal_forest.fit(X_train, y_train)
y_train_pred = optimal_forest.predict(X_train)
train_score = accuracy_score(y_train, y_train_pred)

y_test_pred = optimal_forest.predict(X_test)
test_score=accuracy_score(y_test, y_test_pred)


print('RandomForest:\nin-sample accuracy = %.4f; out-of-sample accuracy = %4f'%(train_score, test_score))


importances = pd.Series(data = optimal_forest.feature_importances_, index = X.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()



####AdaBoost#################
dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 1,random_state = 1)
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=200, random_state=1)

ada.fit(X_train, y_train)

y_train_pred=ada.predict(X_train)
y_test_pred = ada.predict(X_test)


dt_train_acc = accuracy_score(y_train, y_train_pred)
dt_test_acc = accuracy_score(y_test, y_test_pred)

print ('Ada Boost: \nIn-sample accuracy = %.4f; out-of-sample accuracy = %.4f'%(dt_train_acc,dt_test_acc))


#########Bagging########
tree = DecisionTreeClassifier(criterion='gini',
                              random_state=1,
                              max_depth=None)
bag = BaggingClassifier(base_estimator=tree,    #no max_depth constraints
                        n_estimators=500, 
                        max_samples=1.0, 
                        max_features=1.0, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=1, 
                        random_state=1)

bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train_acc = accuracy_score(y_train, y_train_pred)
bag_test_acc = accuracy_score(y_test, y_test_pred)
print('Bsgging: \nIn-sample accuracy = %.4f; out-of-sample accuracy = %.4f'%(bag_train_acc,bag_test_acc))



print("\nMy name is Jianhao Cui")
print("My NetID is: jianhao3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")