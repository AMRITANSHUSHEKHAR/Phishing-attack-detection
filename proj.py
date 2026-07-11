# LOADING DATA
# importing libraries packages to be used in this project
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading data for dataset
data1= pd.read_csv("urldata.csv")
data1.head()

# FAMILIARIZING WITH DATA
#checking the shape of dataset
data1.shape

#listing all the columns/features of the dataset 
data1.columns

#getting information about the dataset
data1.info()

# VISUALIZING THE DATA
# plotting the data distribution

data1.hist(bins=50, figsize= (15,15))
plt.show()

# DATA PREPROCESSING AND EDA

data1.describe()


# DROPPING THE DOMAIN COLUMN
new_data= data1.drop(['Domain'], axis=1).copy()

#checking the data for null or missing values

new_data.isnull().sum()


# shuffling the rows in the dataset so that when spilling the train and test
new_data = new_data.sample(frac=1).reset_index(drop=True)
new_data.head()

# SPLITTING THE DATA

#sepratating and assigning features and taregt columns to X and Y
Y = new_data['Label']
X=new_data.drop('Label', axis=1)
X.shape, Y.shape

# splitting the dataset into train and test set : 90-10 split

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.9, random_state = 12)
X_train.shape, X_test.shape



# MACHINE LEARNING MODELS AND TRAINING
from sklearn.metrics import accuracy_score

# creating holders to store the model performance result
ML_model = []
acc_train = []
acc_test = []

# function to call for storing the results
def store_result(model,a,b):
    ML_model.append(model)
    acc_train.append(round(a,3))
    acc_test.append(round(b,3))


# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

# instantiate the Model
tree = DecisionTreeClassifier(max_depth = 5)

# fit the model
tree.fit(X_train,Y_train)


# predicting the target value from the model for the samples
Y_test_tree = tree.predict(X_test)
Y_train_tree = tree.predict(X_train)


# computing the accuracy of the model performance
acc_train_tree = accuracy_score(Y_train,Y_train_tree)
acc_test_tree = accuracy_score(Y_test,Y_test_tree)

print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))


#checking the feature improtance in the model
plt.figure(figsize=(10,8))
n_features = X_train.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()


#storing the results
store_result('Decision Tree', acc_train_tree, acc_test_tree)



# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# instantiative the model
forest = RandomForestClassifier(max_depth=5)

# fit the model
forest.fit(X_train, Y_train)


# predicting the target value from the model for the samples
Y_test_forest = forest.predict(X_test)
Y_train_forest = forest.predict(X_train)

# computing the accuracy of the model performance
acc_train_forest = accuracy_score(Y_train,Y_train_forest)
acc_test_forest = accuracy_score(Y_test,Y_test_forest)

print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))

# checking the feature improtance in the model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()

# storing the results
store_result('Random Forest', acc_train_forest, acc_test_forest)

# XGBoost Classification model
from xgboost import XGBClassifier

# instantiate the model
xgb = XGBClassifier(learning_rate=0.4, max_depth=5)

# fit the model
xgb.fit(X_train, Y_train)

#predicting the target value from the model for the samples
Y_test_xgb = xgb.predict(X_test)
Y_train_xgb = xgb.predict(X_train)


# computing the accuracy of the model performance
acc_train_xgb = accuracy_score(Y_train,Y_train_xgb)
acc_test_xgb = accuracy_score(Y_test,Y_test_xgb)

print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))

# storing the results
store_result('XGBoost', acc_train_xgb, acc_test_xgb)

#Support vector machine model
from sklearn.svm import SVC

# instantiate the model
svm = SVC(kernel='linear', C=1.0, random_state=12)
#fit the model
svm.fit(X_train, Y_train)


# predicting the target value from the model for the samples
Y_test_svm = svm.predict(X_test)
Y_train_svm = svm.predict(X_train)

# computing the accuracy of the model performance
acc_train_svm = accuracy_score(Y_train,Y_train_svm)
acc_test_svm = accuracy_score(Y_test,Y_test_svm)

print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))

# storing the results
store_result('SVM', acc_train_svm, acc_test_svm)

#creating dataframe
results = pd.DataFrame({ "ML Mode": ML_model,"Train Accuracy": acc_train,"Test Accuracy": acc_test})
results

#Sorting the dataframe on accuracy
results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)

# save XGBoost model to file
import pickle
pickle.dump(xgb, open("XGBClassifier.pickle.dat","wb"))

# testing the saved model
# load model from file

load_model = pickle.load(open("XGBClassifier.pickle.dat","rb"))
load_model