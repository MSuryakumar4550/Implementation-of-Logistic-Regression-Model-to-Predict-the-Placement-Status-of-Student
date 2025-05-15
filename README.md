# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Dataset

2.Create a Copy of the Original Data

3.Drop Irrelevant Columns (sl_no, salary)

4.Check for Missing Values

5.Check for Duplicate Rows

6.Encode Categorical Features using Label Encoding

7.Split Data into Features (X) and Target (y)

8.Split Data into Training and Testing Sets

9.Initialize and Train Logistic Regression Model

10.Make Predictions on Test Set

11.Evaluate Model using Accuracy Score

12.Generate and Display Confusion Matrix

13.Generate and Display Classification Report

14.Make Prediction on a New Sample Input

## Program:
/*
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M.Suryakumar
RegisterNumber:  212224040340

import pandas as pd
pf=pd.read_csv("Placement_Data.csv")
pf.head()
pf1=pf.copy()
pf1=pf1.drop(['sl_no','salary'],axis=1)
pf1.head()
pf1.isnull().sum()
pf1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
pf1["gender"]=le.fit_transform(pf1["gender"])
pf1["ssc_b"]=le.fit_transform(pf1["ssc_b"])
pf1["hsc_b"]=le.fit_transform(pf1["hsc_b"])
pf1["hsc_s"]=le.fit_transform(pf1["hsc_s"])
pf1["degree_t"]=le.fit_transform(pf1["degree_t"])
pf1["workex"]=le.fit_transform(pf1["workex"])
pf1["specialisation"]=le.fit_transform(pf1["specialisation"])
pf1["status"]=le.fit_transform(pf1["status"])
pf1
x=pf1.iloc[:,:-1]
x
y=pf1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
confusion=confusion_matrix(y_test,y_pred)
confusion
classification=classification_report(y_test,y_pred)
print(classification)
lr.predict([[1,80,1,9,1,1,90,1,0,85,1,85]])
*/
```
## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![image](https://github.com/user-attachments/assets/c061fc2c-651d-4c23-944f-454fe3a6157b)
![image](https://github.com/user-attachments/assets/cc3f7687-e046-4556-a75d-9b9c9d7cb0db)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
