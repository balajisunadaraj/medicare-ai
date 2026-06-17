import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Replace the path below with the correct path to your CSV if it's elsewhere
df=pd.read_csv("heart.csv")

mv1 = df.isnull().sum()
print(mv1)


from sklearn.preprocessing import LabelEncoder
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
print(df.dtypes)


from sklearn.model_selection import train_test_split
print("\tData splitting")
x = df.drop('HeartDisease', axis = 1)
print(x)
y = df['HeartDisease']  
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
best_model = trained_models[best_model_name]

with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("Model saved successfully!")

import os
print("Saved model at:", os.path.abspath("best_model.pkl"))

# Optional interactive test at the end; you can remove if running as a script
if __name__ == '__main__':
    try:
        with open('model.pkl','wb') as file: 
            pickle.dump(best_model, file)
    except Exception:
        pass
    print(f"Best model saved to best_model.pkl")
