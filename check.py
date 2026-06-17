import pickle
import os

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model Type:", type(model))

if hasattr(model, "n_features_in_"):
    print("Expected Features:", model.n_features_in_)

print("Model Path:", os.path.abspath("best_model.pkl"))

from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("heart.csv")

for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    print(col)
    print(dict(zip(le.classes_, le.transform(le.classes_))))
    print()