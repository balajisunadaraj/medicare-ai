import os
import pickle
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Lung Cancer Dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "cancer_best_model.pkl")

LUNG_CANCER_FEATURES = [
    "AGE",
    "GENDER",
    "SMOKING",
    "FINGER_DISCOLORATION",
    "MENTAL_STRESS",
    "EXPOSURE_TO_POLLUTION",
    "LONG_TERM_ILLNESS",
    "ENERGY_LEVEL",
    "IMMUNE_WEAKNESS",
    "BREATHING_ISSUE",
    "ALCOHOL_CONSUMPTION",
    "THROAT_DISCOMFORT",
    "OXYGEN_SATURATION",
    "CHEST_TIGHTNESS",
    "FAMILY_HISTORY",
    "SMOKING_FAMILY_HISTORY",
    "STRESS_IMMUNE",
]

TARGET_COLUMN = "PULMONARY_DISEASE"


def load_lung_cancer_dataset(dataset_path=DATASET_PATH):
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=LUNG_CANCER_FEATURES + [TARGET_COLUMN]).copy()
    for feature in LUNG_CANCER_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna(subset=LUNG_CANCER_FEATURES + [TARGET_COLUMN])
    return df


def train_and_save_model(dataset_path=DATASET_PATH, model_path=MODEL_PATH):
    df = load_lung_cancer_dataset(dataset_path)
    x = df[LUNG_CANCER_FEATURES]
    y = df[TARGET_COLUMN].astype(str).str.upper()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    trained_models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    scores = {}
    for name, model in trained_models.items():
        model.fit(x_train, y_train)
        scores[name] = accuracy_score(y_test, model.predict(x_test))

    best_model_name = max(scores, key=scores.get)
    best_model = trained_models[best_model_name]

    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)

    return best_model_name, scores


def predict_from_values(values, model_path=MODEL_PATH):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    cancer_input = pd.DataFrame([values], columns=LUNG_CANCER_FEATURES)
    return model.predict(cancer_input)[0]


if __name__ == "__main__":
    best_model_name, scores = train_and_save_model()
    print("Cancer model trained from Lung Cancer Dataset.csv")
    for model_name, score in scores.items():
        print(f"{model_name}: {score * 100:.2f}%")
    print(f"Best Model: {best_model_name}")
    print(f"Saved to: {MODEL_PATH}")
