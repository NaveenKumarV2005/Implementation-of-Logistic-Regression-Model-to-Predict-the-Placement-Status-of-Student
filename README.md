# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook 

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Naveen Kumar V
RegisterNumber:  212223220068
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("Placement_Data.csv")
print(data.head(), "\n")

print(data["salary"].head(), "\n")

print(data.isnull().sum(), "\n")

print(data.duplicated().sum(), "\n")

data1 = data.drop(["sl_no", "salary"], axis=1)
print(data1.head(), "\n")

le = LabelEncoder()
for col in ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']:
    data1[col] = le.fit_transform(data1[col])

x = data1.iloc[:, :-1]
y = data1["status"]

print(y.value_counts(), "\n")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print(y_pred, "\n")

accuracy = accuracy_score(y_test, y_pred)
print(accuracy, "\n")

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix, "\n")

report = classification_report(y_test, y_pred)
print(report)

sample = pd.DataFrame([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]], columns=x.columns)
prediction = lr.predict(sample)
print("s12")
print(prediction)

```

## Output:
![Screenshot 2025-03-15 141019](https://github.com/user-attachments/assets/2e39fd39-8b06-4d34-aa0d-5a652b143bd6)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
