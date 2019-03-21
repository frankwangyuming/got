h# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:25:46 2019

@author: frank
"""

# Importing new libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split 
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve 
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.filterwarnings('ignore')

## 1. Data preprocessing


death_preds  = pd.read_excel('GOT_character_predictions.xlsx')
death_preds  = pd.DataFrame(death_preds)

## Data visualization

death_preds .shape

death_preds .columns

death_preds .head()
death_preds .info()

## NA values

print ('dataset ({} rows) null value:\n'.format(death_preds .shape[0]))
print (death_preds .isnull().sum(axis = 0))

## mean age
print(death_preds['age'].mean())

## Check which characters have a negative age and it's value.
print(death_preds['name'][death_preds['age']< 0])
print(death_preds['age'][death_preds['age']< 0])

## According the research, Rhaego's age is 0, Doreah is 25
## Replace negative age

death_preds.loc[110, 'age'] = 0.0
death_preds.loc[1350, 'age'] = 25.0

## Check Age value
death_preds['age'].mean()


# Fill the nans 
death_preds["age"].fillna(death_preds["age"].mean(), inplace=True)
death_preds["culture"].fillna("", inplace=True)

# Get all of the culture values in our dataset
set(death_preds['culture'])

## culture split and combine
cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Andal': ['andal', 'andals'],
    'Lysene': ['lysene', 'lyseni'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Vale': ['vale', 'valemen', 'vale mountain clans'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Mereen': ['meereen', 'meereenese'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Reach': ['the reach', 'reach', 'reachmen'],
    'Qartheen': ['qartheen', 'qarth'],
    'Ironborn': ['ironborn', 'ironmen'],
    'RiverLands': ['riverlands', 'rivermen']
}

def combine_culture(value):
    value = value.lower()
    i = [j for (j, i) in cult.items() if value in i]
    return i[0] if len(i) > 0 else value.title()
death_preds.loc[:, "culture"] = [combine_culture(x) for x in death_preds["culture"]]

## fill missing value
death_preds.loc[:, "title"] = pd.factorize(death_preds.title)[0]
death_preds.loc[:, "culture"] = pd.factorize(death_preds.culture)[0]
death_preds.loc[:, "mother"] = pd.factorize(death_preds.mother)[0]
death_preds.loc[:, "father"] = pd.factorize(death_preds.father)[0]
death_preds.loc[:, "heir"] = pd.factorize(death_preds.heir)[0]
death_preds.loc[:, "house"] = pd.factorize(death_preds.house)[0]
death_preds.loc[:, "spouse"] = pd.factorize(death_preds.spouse)[0]

death_preds.fillna(value = -1, inplace = True)

## feature engineer
death_preds.drop(["name","dateOfBirth","S.No"], 1, inplace = True)
death_preds.columns = map(lambda x: x.replace(".", "").replace("_", ""), death_preds.columns)

## Corr visualization
sns.heatmap(death_preds.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(30,20)
plt.show()

## 2. Data Analysis

## 2.1
f,ax=plt.subplots(2,2,figsize=(17,15))
sns.violinplot("isNoble", "isAliveMother", hue="isAlive", data=death_preds ,split=True, ax=ax[0, 0])
ax[0, 0].set_title('isNoble and isAliveMother vs Mortality')
ax[0, 0].set_yticks(range(2))

sns.violinplot("isNoble", "male", hue="isAlive", data=death_preds ,split=True, ax=ax[0, 1])
ax[0, 1].set_title('isNoble and Male vs Mortality')
ax[0, 1].set_yticks(range(2))

sns.violinplot("isNoble", "isMarried", hue="isAlive", data=death_preds ,split=True, ax=ax[1, 0])
ax[1, 0].set_title('isNoble and isMarried vs Mortality')
ax[1, 0].set_yticks(range(2))


sns.violinplot("isNoble", "book1AGameOfThrones", hue="isAlive", data=death_preds ,split=True, ax=ax[1, 1])
ax[1, 1].set_title('isNoble and book1AGameOfThrones vs Mortality')
ax[1, 1].set_yticks(range(2))

plt.show()

##2.2
f,ax=plt.subplots(2,2,figsize=(17,15))
sns.violinplot("isNoble", "book2AClashOfKings", hue="isAlive", data=death_preds ,split=True, ax=ax[0, 0])
ax[0, 0].set_title('isNoble and book2AClashOfKings vs Mortality')
ax[0, 0].set_yticks(range(2))

sns.violinplot("isNoble", "book3AStormOfSwords", hue="isAlive", data=death_preds ,split=True, ax=ax[0, 1])
ax[0, 1].set_title('isNoble and book3AStormOfSwords vs Mortality')
ax[0, 1].set_yticks(range(2))

sns.violinplot("isNoble", "book4AFeastForCrows", hue="isAlive", data=death_preds ,split=True, ax=ax[1, 0])
ax[1, 0].set_title('isNoble and book4AFeastForCrows vs Mortality')
ax[1, 0].set_yticks(range(2))


sns.violinplot("isNoble", "book5ADancewithDragons", hue="isAlive", data=death_preds ,split=True, ax=ax[1, 1])
ax[1, 1].set_title('isNoble and book5ADancewithDragons vs Mortality')
ax[1, 1].set_yticks(range(2))

plt.show()

## 3. GOT modeling

## create dummy variables 
df = pd.get_dummies(death_preds)

x = df.iloc[:,0:-1]
y = df.iloc[:,-1:]

## import new library
from sklearn.feature_selection import RFE # recursive feature elimination 
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

##Select the top 15 most important features
rfe = RFE(logreg, 15) 
rfe = rfe.fit(x, y )
print(rfe.support_)
print(rfe.ranking_)

## feature choose

cols = ['male','mother', 'father', 'heir','book1AGameOfThrones', 'book2AClashOfKings',
       'book3AStormOfSwords', 'book4AFeastForCrows','isAliveMother','isAliveFather', 
       'isAliveSpouse','isMarried', 'isNoble', 'numDeadRelations','popularity']

X=death_preds[cols]

## Implementing the model
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())

## 4. Logistic Regression Model Fitting

## 4.1 split the data
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(),
                                                    test_size=0.1, random_state=508,
                                                    stratify = y)

## 4.2 fitting
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

## 4.3 prediction
y_pred = logreg.predict(X_test)

## 4.4 Accuracy
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

## Cross-validation
kfold = model_selection.KFold(n_splits=3, random_state=508)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("3-fold cross validation average accuracy: %.3f" % (results.mean()))

cv_lr_3 = cross_val_score(logreg, X, y, cv = 3)

print(pd.np.mean(cv_lr_3).round(3))

## 5. Confusion Matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))

## 6. ROC 

logit_roc_auc1 = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc1)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GOT ROC prediction')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

print('AUC is',logit_roc_auc1)

## 7 Random Forest fitting, prediction and AUC
rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 16,
                                    n_estimators = 600,
                                    warm_start = True)



rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))

## 7.1AUC
logit_roc_auc2 = roc_auc_score(y_test, rf_optimal_pred).round(3)

print('AUC is',logit_roc_auc2)


## Gradient Boosted fitting, prediction and AUC

gbm_optimal = GradientBoostingClassifier(criterion = 'friedman_mse',
                                      learning_rate = 0.1,
                                      max_depth = 5,
                                      n_estimators = 100,
                                      random_state = 508)



gbm_optimal.fit(X_train, y_train)


gbm_optimal_score = gbm_optimal.score(X_test, y_test)


gbm_optimal_pred = gbm_optimal.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('Testing Score:', gbm_optimal.score(X_test, y_test).round(4))

## 7.2AUC
logit_roc_auc3 = roc_auc_score(y_test, gbm_optimal_pred).round(3)

print('AUC is',logit_roc_auc3)


## 9. Predicting the death of characters by different models

## 9.1 Random Forest
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

print('RandomForest Accuracy：(original)\n',random_forest.score(X_train, y_train))

## 9.2 Decision Tree 
Dtree=DecisionTreeClassifier()

Dtree.fit(X_train,y_train)

print('DecisionTree Accuracy：(original)\n',Dtree.score(X_train, y_train))

## 9.3 SVC
svc = SVC()

svc.fit(X_train, y_train)

print('SVC Accuracy：\n',svc.score(X_train, y_train))

## 9.4 KNN
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

print('kNN Accuracy：\n',knn.score(X_train, y_train))

## 9.5 NaiveBayes Gaussian
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

print('gaussian Accuracy：\n',gaussian.score(X_train, y_train))

## 9.6 Cross validated score for Gradient Boosting
grad=GradientBoostingClassifier(n_estimators=500,random_state=508,learning_rate=0.1)

result=cross_val_score(grad,X_train,y_train,cv=10,scoring='accuracy')

print('The cross validated score for Gradient Boosting is:',result.mean())

## 10. Visualizing the tree

import graphviz
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 508,
                                        max_depth = 3,
                                        min_samples_leaf = 10)
dtree.fit(X_train,y_train)

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names = X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

## 11. Feature Importance

df2 = df.copy(deep=True)

x = df2.iloc[:,0:-1].values
y = df2.iloc[:,-1:].values

df2.drop(["isAlive"], inplace=True, axis=1)

rf_clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=508)

rf_clf.fit(x,y)

# Plot the 15 most important features
plt.figure()
pd.Series(rf_clf.feature_importances_, 
          df2.columns).sort_values(ascending=True)[15:].plot.barh(width=0.5,ax=plt.gca())
plt.gca().set_title('Random Forest Feature Importance')


# Saving model predictions

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'RF_Predicted': rf_optimal_pred,
                                     'GBM_Predicted': gbm_optimal_pred})


model_predictions_df.to_excel("Ensemble_Model_Predictions.xlsx")











