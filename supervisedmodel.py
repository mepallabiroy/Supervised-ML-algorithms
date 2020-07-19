#Code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
dataset = pd.read_csv("train_AV3.csv")
print(dataset)
dataset.dtypes
x = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]
print(x)
print(y)
dataset.columns[dataset.isnull().any()]
dataset["Gender"].fillna("0",inplace = True)
dataset["Married"].fillna("0",inplace = True)
dataset["Dependents"].fillna("0",inplace = True)
dataset["Self_Employed"].fillna("0",inplace = True)
dataset["LoanAmount"].fillna("0",inplace = True)
dataset["Loan_Amount_Term"].fillna("0",inplace = True)
dataset["Credit_History"].fillna("0",inplace = True)
dataset.columns[dataset.isnull().any()]
x_encoded = LabelEncoder()
dataset["encoded_gender"] = x_encoded.fit_transform(dataset["Gender"])
dataset["encoded_married"] = x_encoded.fit_transform(dataset["Married"])
dataset["encoded_education"] = x_encoded.fit_transform(dataset["Education"])
dataset["encoded_property_area"] = x_encoded.fit_transform(dataset["Property_Area"])
dataset["encoded_self_employed"] = x_encoded.fit_transform(dataset["Self_Employed"])
print(dataset)
dataset.drop(["Gender", "Married", "Education", "Property_Area", "Self_Employed"], axis=1, inplace=True)
dataset["Dependents"].replace("3+","3", inplace=True)
print(dataset)
x_new = dataset.drop(["Loan_ID", "Loan_Status"], axis=1)
y_new = dataset["Loan_Status"]
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size = 0.2, random_state = 1)
accuracy = {}

svm = SVC(kernel= "rbf", degree=2, gamma="auto", verbose = True, random_state =1)
svm.fit(x_train, y_train)
y_predict_svm = svm.predict(x_test)
confusion_matrix(y_test,y_predict_svm)
accuracy_score(y_test, y_predict_svm)
accuracy["SVM"]=accuracy_score(y_test, y_predict_svm)

regressor = LogisticRegression(multi_class="ovr",random_state = 1)
regressor.fit(x_train, y_train)
y_predict_lr = regressor.predict(x_test)
confusion_matrix(y_test, y_predict_lr)
accuracy_score(y_test, y_predict_lr)
accuracy["Logistic Regression"] = accuracy_score(y_test, y_predict_lr)
regressor.predict_proba(x_test)

cv = KFold(n_splits=3, shuffle=True, random_state=1)
accuracy_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x_new,dataset['Loan_Status'],cv=cv)
    accuracy_rate.append(score.mean())
 error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x_new,dataset['Loan_Status'],cv=cv)
    error_rate.append(1-score.mean())
print(accuracy_rate)
print(error_rate)
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn_model = KNeighborsClassifier(n_neighbors=35)
knn_model.fit(x_train,y_train)
y_predict_knn = knn_model.predict(x_test)
confusion_matrix(y_test,y_predict_knn)
accuracy_score(y_test,y_predict_knn)
accuracy["KNN"] = accuracy_score(y_test, y_predict_knn)

decisiontree = DecisionTreeClassifier(criterion = "entropy", random_state= 1)
decisiontree.fit(x_train, y_train)
y_predict_tree = decisiontree.predict(x_test)
accuracy_score(y_test,y_predict_tree)
accuracy["Decision Tree"] = accuracy_score(y_test, y_predict_tree)
confusion_matrix(y_test, y_predict_tree)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred_nb = gnb.predict(x_test)
confusion_matrix(y_test, y_pred_nb)
accuracy_score(y_test, y_pred_nb)
accuracy["Naive Bayes"] = accuracy_score(y_test, y_pred_nb)

from sklearn.ensemble import RandomForestClassifier
accuracy_rate = []

for i in range(1,40):
    
    rand_forest = RandomForestClassifier(n_estimators=i)
    score=cross_val_score(rand_forest,x_new,dataset['Loan_Status'],cv=cv)
    accuracy_rate.append(score.mean())
error_rate = []

for i in range(1,40):
    
    rand_forest = RandomForestClassifier(n_estimators=i)
    score=cross_val_score(rand_forest,x_new,dataset['Loan_Status'],cv=cv)
    error_rate.append(1 - score.mean())
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='*',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. No. of estimators')
plt.xlabel('K')
plt.ylabel('Error Rate')    

rand_forest_model = RandomForestClassifier(n_estimators = 35)
rand_forest_model.fit(x_train, y_train)
y_pred_rf = rand_forest_model.predict(x_test)
confusion_matrix(y_test, y_pred_rf)
accuracy_score(y_test, y_pred_rf)
accuracy["RandomForest"] = accuracy_score(y_test, y_pred_rf)

print(accuracy)
print(max(accuracy, key = accuracy.get))
print(min(accuracy, key = accuracy.get))
