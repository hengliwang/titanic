import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
# Going to use these 5 base models for the stacking
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


train = pd.read_csv('train_result.csv')
test = pd.read_csv('test_result.csv')

PassengerId = test['PassengerId']
train.drop(['PassengerId'],axis=1)
test.drop(['PassengerId'],axis=1)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

#mode = LogisticRegression();
#mode.fit(x_train,y_train);
#prediction = mode.predict(x_test);
#pipline = Pipeline([('df',RandomForestClassifier())])
grids=GridSearchCV(RandomForestClassifier(), param_grid=[{'n_estimators':[100,200,400,500,600,800,100],'max_depth':[1,2,4,6,8,10,12],'min_samples_leaf': [1,2,4,6,8,10,12]}])
#训练以及调参
grids.fit(x_train, y_train)

#rf = RandomForestClassifier(n_estimators=1000,criterion="gini",max_depth=6,min_samples_split=2);
#rf.fit(x_train,y_train)
prediction = grids.predict(x_test)



StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': prediction })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)