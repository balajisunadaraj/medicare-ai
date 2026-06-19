import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("The_Cancer_data_1500_V2.csv")

mv1 = df.isnull().sum()
print(mv1)

from sklearn.preprocessing import LabelEncoder
categorical_cols = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
print(df.dtypes)

from sklearn.model_selection import train_test_split
print("\tData splitting")
x = df.drop('Diagnosis', axis=1)
print(x)
y = df['Diagnosis']  
print(y)
print('train test splitting')
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.2)
print("total number of data:",len(df))
print("total number of test data:",len(x_test))
print("Total number of train data : ", len(x_train))

from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model=RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model.fit(x_train,y_train)
y_predit=model.predict(x_test)
print(y_predit)
acc=accuracy_score(y_test,y_predit)

from sklearn.linear_model import LogisticRegression
m2=LogisticRegression(max_iter=200)
m2.fit(x_train,y_train.values.ravel())
y_prediction_log=m2.predict(x_test)
log_acc=accuracy_score(y_test, y_prediction_log)

from sklearn.tree import DecisionTreeClassifier
m3=DecisionTreeClassifier(random_state=42)
m3.fit(x_train,y_train.values.ravel())
y_predit=m3.predict(x_test)
data_acc=accuracy_score(y_test,y_predit)

import matplotlib.pyplot as plt

models = {
    "Random Forest": acc,
    "Logistic Regression": log_acc,
    "Decision Tree": data_acc
}


trained_models = {
    "Random Forest": model,
    "Logistic Regression": m2,
    "Decision Tree": m3
}

print(f"Random Forest Accuracy      : {acc*100:.2f}%")
print(f"Logistic Regression Accuracy: {log_acc*100:.2f}%")
print(f"Decision Tree Accuracy      : {data_acc*100:.2f}%")

best_model_name = max(models, key=models.get)
print("Best Model:", best_model_name)

import pickle
cancer_best_model = trained_models[best_model_name]

with open("cancer_best_model.pkl", "wb") as file:
    pickle.dump(cancer_best_model, file)

print("Model saved successfully!")

import pickle
print(f"clf accuracy: {acc*100:.2f}%")
with open('model.pkl','wb') as file: 
    pickle.dump(cancer_best_model, file)
with open('model.pkl','rb') as file:
    loaded_model=pickle.load(file)
a=float(input("enter Age:"))
b=float(input("enter Gender:"))
c=float(input("enter BMI:"))
d=float(input("enter Smoking:"))
e=float(input("enter GeneticRisk:"))
f=float(input("enter PhysicalActivity:"))
g=float(input("enter AlcoholIntake:"))
h=float(input("enter CancerHistory:"))
final=[[a,b,c,d,e,f,g,h]]
prediction = loaded_model.predict(final)
print(prediction)

print("Best model accuracy")

result=[[a,b,c,d,e,f,g,h]]
best_mod = loaded_model.predict(result)
print(best_mod)