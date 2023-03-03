import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
 
#load the training data and prepare X and y
Data = pd.read_csv("train.csv",header=None)
print('Dimension of Training Data ( Records: ',Data.shape[0],', Columns: ',Data.shape[1],')')

X = Data.drop([Data.columns[-1]], axis = 1)
y = Data[Data.columns[-1]]

print('Dimension of X: ',X.shape)
print('Dimension of y: ',y.shape)

#load the test data
tstData = pd.read_csv("test.csv",header=None)
print('\nDimension of Test Data ( Records: ',tstData.shape[0],', Columns: ',tstData.shape[1],')')

X_test = tstData.drop([tstData.columns[-1]], axis = 1)
y_test = tstData[tstData.columns[-1]]

print('Dimension of X: ',X_test.shape)
print('Dimension of y: ',y_test.shape)

print('\nClass imbalance: ', np.unique(y, return_counts=True)[1])

# Then split the training dataset into training, validation
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

#creat the xgb model
xgb = XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=-1)
#xgb.fit(X, y)
#scores = cross_val_score(xgb, X, y, cv=5, scoring='accuracy')
#print('5-CV accuracy:', "{0:.5f}".format(np.mean(scores)))
#print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))
#print(xgb.get_params())

# Create the parameter grid
params = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=500, num=5)],
    'max_depth': [i for i in range(3, 10)],
    'min_child_weight': [i for i in range(1, 7)],
    'subsample': [i/10.0 for i in range(6,11)],
    'colsample_bytree': [i/10.0 for i in range(6,11)]
}

# Create the randomised grid search model
rgs = RandomizedSearchCV(estimator=xgb, param_distributions=params, n_iter=20, cv=5, random_state=42, n_jobs=-1,
        scoring='accuracy', return_train_score=True, verbose=3)

#fit the rgm model
rgs.fit(X, y)

# Evaluate best models on the test set
best_xgb = rgs.best_estimator_

predictions = best_xgb.predict(X_test)
y_pred = [round(value) for value in predictions]

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#store the scores to file
fp = open('score.txt','w')
fp.write(str(accuracy))
fp.close()

#save the final model
filename = 'bestmodel.sav'
pickle.dump(best_xgb, open(filename, 'wb'))

#how to load and evaluate in future
#model = pickle.load(open(filename, 'rb'))
