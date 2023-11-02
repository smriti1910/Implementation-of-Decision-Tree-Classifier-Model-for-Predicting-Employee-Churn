# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use standard python library
2. Set dataset values
3. Import linear regression
4. Predict values and calculate accuracy
5. Obtain a graph i.e. visualization of regression line


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Smriti .B
RegisterNumber:  212221040156
*/
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
data["Departments "]=le.fit_transform(data["Departments "])
data.head()
data.info()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

x_train.shape
x_test.shape
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

print(y_pred)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])


## Output:
Data head
![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/f73a6b6a-dc8d-46b9-bec0-925f2987b8b4)

Data info
![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/fdbd28e8-5c14-4dcc-94b2-a4f2f116dac4)

Null dataset
![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/3e00b52b-efa3-4fe2-9318-d76b368eb014)

Values count in left column
![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/021f53bb-329c-46a3-987e-5b5b69d65b6f)

Dataset transformed head
![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/82c38a2a-c053-4fb9-8ce2-aa0a54f41817)

x.head
![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/4128b7da-8aa6-4224-bd12-4623e0cdb4fa)

![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/ed4abbf1-f5bd-4c6e-85ce-fbf43d72326f)

![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/d5c6c6f2-0adb-4ff7-b269-9abb660a708e)

![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/13872b4b-f536-4593-8a7c-5c2c27d77038)

![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/b2c9fa75-ce3e-4fcf-b16f-89732c9b04e4)

![image](https://github.com/smriti1910/Ex-06---Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/133334803/d49e4ec1-76e9-4faa-b20b-27a994b5849f)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
