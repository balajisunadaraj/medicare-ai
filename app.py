import os
import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from auth import init_auth, login_manager, UserMixin, login_user, logout_user, current_user
from database import db, init_db
from werkzeug.security import generate_password_hash, check_password_hash
from Cancer_prediction_dataset import LUNG_CANCER_FEATURES

app = Flask("Medicare")
app.secret_key = "Medicare-ai@123#"

basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "MEDICARE_DATABASE_URI",
    f"sqlite:///{os.path.join(basedir, 'medicare.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

init_auth(app)
init_db(app)
MODEL_PATH = os.path.join(basedir, "best_model.pkl")
CANCER_MODEL_PATH = os.path.join(basedir, "cancer_best_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

cancer_model = None
if os.path.exists(CANCER_MODEL_PATH):
    with open(CANCER_MODEL_PATH, "rb") as f:
        cancer_model = pickle.load(f)

FEATURES = [
    {
        "title": "Heart Risk Assessment",
        "description": "Enter your health details and get an instant prediction based on medical risk factors.",
        "icon": "❤️"
    },
    {
        "title": "Smooth & Responsive",
        "description": "Responsive layout with smooth scrolling, subtle animations, and an intuitive user experience.",
        "icon": "⚡"
    },
    {
        "title": "Patient Education",
        "description": "Learn about signs, symptoms, and prevention tips for heart disease in one polished dashboard.",
        "icon": "📘"
    }
]

INSTRUCTIONS = [
    "Provide accurate values for your age, blood pressure, cholesterol, and exercise metrics.",
    "Use the dropdown options for encoded medical categories like chest pain type and resting ECG.",
    "Submit the form to receive an instant AI-based heart disease risk prediction.",
    "If you are unsure about any value, consult a healthcare professional before relying on the result."
]

PRECAUTIONS = [
    "If you experience chest pain, shortness of breath, or dizziness, seek immediate medical attention.",
    "Share results with a qualified doctor before making any health decisions.",
    "Maintain regular checkups, a healthy diet, and consistent exercise for better heart health."
]

CANCER_INSTRUCTIONS = [
    "Provide truthful lung health, symptom, exposure, and family-history details.",
    "Use Yes or No fields for symptoms and risk factors, and numeric values for age, energy level, and oxygen saturation.",
    "Submit the form to receive an estimate from the trained lung cancer prediction model.",
    "Review results carefully and seek medical advice rather than relying on this prediction alone."
]

CANCER_PRECAUTIONS = [
    "A predictive model is not a diagnostic tool; consult a healthcare professional for any concerns.",
    "Seek prompt care for persistent cough, chest pain, shortness of breath, throat discomfort, or low oxygen saturation.",
    "Avoid tobacco smoke and pollution exposure where possible, and keep follow-up screenings as recommended by your doctor."
]


def parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def predict_heart_disease(form):
    payload = [
        parse_int(form.get("AGE", 0)),
        parse_int(form.get("GENDER", 0)),
        parse_int(form.get("SMOKING", 0)),
        parse_int(form.get("FINGER_DISCOLORATION", 0)),
        parse_int(form.get("MENTAL_STRESS", 0)),
        parse_int(form.get("EXPOSURE_TO_POLLUTION", 0)),
        parse_int(form.get("LONG_TERM_ILLNESS", 0)),
        parse_float(form.get("ENERGY_LEVEL", 0)),
        parse_int(form.get("IMMUNE_WEAKNESS", 0)),
        parse_int(form.get("BREATHING_ISSUE", 0.0)),
        parse_int(form.get("ALCOHOL_CONSUMPTION", 0)),
        parse_int(form.get("THROAT_DISCOMFORT", 0)),
        parse_float(form.get("OXYGEN_SATURATION", 0)),
        parse_int(form.get("CHEST_TIGHTNESS", 0)),
        parse_int(form.get("FAMILY_HISTORY", 0)),
        parse_int(form.get("SMOKING_FAMILY_HISTORY", 0)),
        parse_int(form.get("STRESS_IMMUNE", 0)),
    ]

    if model is not None:
        prediction = model.predict([payload])
        try:
            prob = model.predict_proba([payload])[0]
            risk_prob = int(prob[1] * 100) if len(prob) > 1 else 50
            no_risk_prob = 100 - risk_prob
        except:
            risk_prob = 50
            no_risk_prob = 50
        
        if int(prediction[0]) == 1:
            return {
                "label": "Heart Disease Detected",
                "message": "The model predicts a likely presence of heart disease. Please consult a medical professional.",
                "class": "risk-high",
                "probability": {"risk": risk_prob, "no_risk": no_risk_prob}
            }
        return {
            "label": "No Heart Disease Detected",
            "message": "The model predicts no heart disease. Keep monitoring your health and maintain healthy habits.",
            "class": "risk-low",
            "probability": {"risk": risk_prob, "no_risk": no_risk_prob}
        }

    return {
        "label": "Model unavailable",
        "message": "No trained model was found. Place best_model.pkl in the project root to enable real predictions.",
        "class": "risk-medium",
        "probability": {"risk": 50, "no_risk": 50}
    }


def calculate_bmi(weight, height):
    try:
        height_m = float(height) / 100.0
        if height_m <= 0:
            return 0.0
        return round(float(weight) / (height_m ** 2), 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def build_health_dashboard(form):
    age = parse_int(form.get("age", 0))
    weight = parse_float(form.get("weight", 0.0))
    height = parse_float(form.get("height", 0.0))
    resting_bp = parse_int(form.get("resting_bp", 0))
    cholesterol = parse_int(form.get("cholesterol", 0))
    fasting_bs = parse_int(form.get("fasting_bs", 0))
    exercise_minutes = parse_int(form.get("exercise_minutes", 0))
    sleep_hours = parse_int(form.get("sleep_hours", 7))
    stress_level = parse_int(form.get("stress_level", 5))

    bmi = calculate_bmi(weight, height)
    risk_points = 0

    if age >= 55:
        risk_points += 2
    elif age >= 45:
        risk_points += 1

    if resting_bp >= 140:
        risk_points += 2
    elif resting_bp >= 130:
        risk_points += 1

    if cholesterol >= 240:
        risk_points += 2
    elif cholesterol >= 200:
        risk_points += 1

    if fasting_bs == 1:
        risk_points += 2

    if exercise_minutes < 150:
        risk_points += 2
    elif exercise_minutes < 210:
        risk_points += 1

    if sleep_hours < 6 or sleep_hours > 9:
        risk_points += 1

    if stress_level >= 8:
        risk_points += 2
    elif stress_level >= 6:
        risk_points += 1

    if bmi >= 30:
        risk_points += 2
    elif bmi >= 25:
        risk_points += 1

    if risk_points <= 3:
        risk_level = "Low"
        advice = "Your health profile indicates low risk. Keep up healthy habits and routine checkups."
    elif risk_points <= 6:
        risk_level = "Moderate"
        advice = "Moderate risk detected. Improve diet, exercise regularly, and follow up with a doctor."
    else:
        risk_level = "High"
        advice = "High risk detected. Please seek medical advice and consider lifestyle changes quickly."

    return {
        "bmi": bmi,
        "risk_level": risk_level,
        "risk_points": risk_points,
        "advice": advice,
        "metrics": [
            {"label": "BMI", "value": f"{bmi}", "note": "Body Mass Index."},
            {"label": "Cholesterol", "value": f"{cholesterol} mg/dL", "note": "Lower is better."},
            {"label": "Blood Pressure", "value": f"{resting_bp} mm Hg", "note": "Keep within healthy range."},
            {"label": "Weekly Activity", "value": f"{exercise_minutes} min", "note": "Target 150+ min per week."}
        ]
    }


CANCER_FEATURE_NAMES = LUNG_CANCER_FEATURES

CANCER_FIELD_LABELS = {
    "AGE": "Age",
    "GENDER": "Gender",
    "SMOKING": "Currently Smoking",
    "FINGER_DISCOLORATION": "Finger Discoloration",
    "MENTAL_STRESS": "Mental Stress",
    "EXPOSURE_TO_POLLUTION": "Pollution Exposure",
    "LONG_TERM_ILLNESS": "Long-Term Illness",
    "ENERGY_LEVEL": "Energy Level",
    "IMMUNE_WEAKNESS": "Immune Weakness",
    "BREATHING_ISSUE": "Breathing Issue",
    "ALCOHOL_CONSUMPTION": "Alcohol Consumption",
    "THROAT_DISCOMFORT": "Throat Discomfort",
    "OXYGEN_SATURATION": "Oxygen Saturation",
    "CHEST_TIGHTNESS": "Chest Tightness",
    "FAMILY_HISTORY": "Family History",
    "SMOKING_FAMILY_HISTORY": "Smoking in Family History",
    "STRESS_IMMUNE": "Stress-Related Immune Issues",
}

CANCER_FORM_DEFAULTS = {
    "AGE": "45",
    "GENDER": "0",
    "SMOKING": "0",
    "FINGER_DISCOLORATION": "0",
    "MENTAL_STRESS": "0",
    "EXPOSURE_TO_POLLUTION": "0",
    "LONG_TERM_ILLNESS": "0",
    "ENERGY_LEVEL": "60",
    "IMMUNE_WEAKNESS": "0",
    "BREATHING_ISSUE": "0",
    "ALCOHOL_CONSUMPTION": "0",
    "THROAT_DISCOMFORT": "0",
    "OXYGEN_SATURATION": "97",
    "CHEST_TIGHTNESS": "0",
    "FAMILY_HISTORY": "0",
    "SMOKING_FAMILY_HISTORY": "0",
    "STRESS_IMMUNE": "0",
}

CANCER_FORM_FIELDS = [
    {
        "name": "AGE",
        "label": "Age",
        "type": "number",
        "min": 1,
        "max": 120,
        "step": 1,
        "help": "Patient age in years.",
    },
    {
        "name": "GENDER",
        "label": "Gender",
        "type": "select",
        "options": [("0", "Female"), ("1", "Male")],
        "help": "Select the gender value used by the dataset.",
    },
    {
        "name": "ENERGY_LEVEL",
        "label": "Energy Level",
        "type": "number",
        "min": 0,
        "max": 100,
        "step": 0.1,
        "help": "Approximate energy score from 0 to 100.",
    },
    {
        "name": "OXYGEN_SATURATION",
        "label": "Oxygen Saturation",
        "type": "number",
        "min": 70,
        "max": 100,
        "step": 0.1,
        "help": "SpO2 percentage, such as 97.",
    },
]

for field_name in [
    "SMOKING",
    "FINGER_DISCOLORATION",
    "MENTAL_STRESS",
    "EXPOSURE_TO_POLLUTION",
    "LONG_TERM_ILLNESS",
    "IMMUNE_WEAKNESS",
    "BREATHING_ISSUE",
    "ALCOHOL_CONSUMPTION",
    "THROAT_DISCOMFORT",
    "CHEST_TIGHTNESS",
    "FAMILY_HISTORY",
    "SMOKING_FAMILY_HISTORY",
    "STRESS_IMMUNE",
]:
    CANCER_FORM_FIELDS.append(
        {
            "name": field_name,
            "label": CANCER_FIELD_LABELS[field_name],
            "type": "select",
            "options": [("0", "No"), ("1", "Yes")],
            "help": "Choose Yes if this factor applies.",
        }
    )


def parse_cancer_feature_value(field_name, value):
    if field_name in ["AGE"]:
        return parse_int(value, parse_int(CANCER_FORM_DEFAULTS[field_name]))
    if field_name in ["ENERGY_LEVEL", "OXYGEN_SATURATION"]:
        return parse_float(value, parse_float(CANCER_FORM_DEFAULTS[field_name]))
    return 1 if str(value).strip() in ["1", "Yes", "yes", "true", "True"] else 0


def get_cancer_probability_index(model):
    classes = list(getattr(model, "classes_", []))
    for risk_label in ["YES", "Yes", "yes", "1", 1, True]:
        if risk_label in classes:
            return classes.index(risk_label)
    return 1 if len(classes) > 1 else 0


def predict_cancer(form):
    payload = [
        parse_cancer_feature_value(feature_name, form.get(feature_name, CANCER_FORM_DEFAULTS[feature_name]))
        for feature_name in CANCER_FEATURE_NAMES
    ]

    risk_prob = 50
    no_risk_prob = 50

    if cancer_model is not None:
        try:
            cancer_input = pd.DataFrame([payload], columns=CANCER_FEATURE_NAMES)
            prediction = cancer_model.predict(cancer_input)
            try:
                prob = cancer_model.predict_proba(cancer_input)[0]
                risk_index = get_cancer_probability_index(cancer_model)
                risk_prob = int(prob[risk_index] * 100) if len(prob) > risk_index else 50
                no_risk_prob = 100 - risk_prob
            except Exception:
                risk_prob = 50
                no_risk_prob = 50
            
            pred_label = str(prediction[0])
            if pred_label.lower() in ["yes", "1", "true"]:
                return {
                    "label": "Lung Cancer Risk Detected",
                    "message": "The model predicts possible lung cancer or pulmonary disease risk. Please consult a qualified healthcare professional.",
                    "class": "risk-high",
                    "probability": {"risk": risk_prob, "no_risk": no_risk_prob}
                }
            return {
                "label": "Low Lung Cancer Risk",
                "message": "The model predicts low lung cancer or pulmonary disease risk based on the provided inputs.",
                "class": "risk-low",
                "probability": {"risk": risk_prob, "no_risk": no_risk_prob}
            }
        except Exception as e:
            return {
                "label": "Prediction Error",
                "message": f"An error occurred during prediction: {str(e)}",
                "class": "risk-medium",
                "probability": {"risk": 50, "no_risk": 50}
            }

    return {
        "label": "Cancer model unavailable",
        "message": "No cancer model file was found. Place cancer_best_model.pkl in the project root to enable cancer risk prediction.",
        "class": "risk-medium",
        "probability": {"risk": 50, "no_risk": 50}
    }


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    resting_bp = db.Column(db.Integer, nullable=False)
    cholesterol = db.Column(db.Integer, nullable=False)
    fasting_bs = db.Column(db.Integer, nullable=False)
    exercise_minutes = db.Column(db.Integer, nullable=False)
    sleep_hours = db.Column(db.Integer, nullable=False)
    stress_level = db.Column(db.Integer, nullable=False)


class PredictionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    AGE = db.Column(db.Integer, nullable=False)
    GENDER = db.Column(db.Integer, nullable=False)
    SMOKING = db.Column(db.Integer, nullable=False)
    FINGER_DISCOLORATION = db.Column(db.Integer, nullable=False)
    MENTAL_STRESS = db.Column(db.Integer, nullable=False)
    EXPOSURE_TO_POLLUTION = db.Column(db.Integer, nullable=False)
    LONG_TERM_ILLNESS = db.Column(db.Integer, nullable=False)
    ENERGY_LEVEL = db.Column(db.Float, nullable=False)
    IMMUNE_WEAKNESS = db.Column(db.Integer, nullable=False)
    BREATHING_ISSUE = db.Column(db.Integer, nullable=False)
    ALCOHOL_CONSUMPTION = db.Column(db.Integer, nullable=False)
    THROAT_DISCOMFORT = db.Column(db.Integer, nullable=False)
    OXYGEN_SATURATION = db.Column(db.Float, nullable=False)
    CHEST_TIGHTNESS = db.Column(db.Integer, nullable=False)
    FAMILY_HISTORY = db.Column(db.Integer, nullable=False)
    SMOKING_FAMILY_HISTORY = db.Column(db.Integer, nullable=False)
    STRESS_IMMUNE = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(64), nullable=False)
    message = db.Column(db.String(256), nullable=False)
    risk_class = db.Column(db.String(32), nullable=False)


class CancerPredictionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    AGE = db.Column(db.Integer, nullable=False)
    GENDER = db.Column(db.Integer, nullable=False)
    SMOKING = db.Column(db.Integer, nullable=False)
    FINGER_DISCOLORATION = db.Column(db.Integer, nullable=False)
    MENTAL_STRESS = db.Column(db.Integer, nullable=False)
    EXPOSURE_TO_POLLUTION = db.Column(db.Integer, nullable=False)
    LONG_TERM_ILLNESS = db.Column(db.Integer, nullable=False)
    ENERGY_LEVEL = db.Column(db.Float, nullable=False)
    IMMUNE_WEAKNESS = db.Column(db.Integer, nullable=False)
    BREATHING_ISSUE = db.Column(db.Integer, nullable=False)
    ALCOHOL_CONSUMPTION = db.Column(db.Integer, nullable=False)
    THROAT_DISCOMFORT = db.Column(db.Integer, nullable=False)
    OXYGEN_SATURATION = db.Column(db.Float, nullable=False)
    CHEST_TIGHTNESS = db.Column(db.Integer, nullable=False)
    FAMILY_HISTORY = db.Column(db.Integer, nullable=False)
    SMOKING_FAMILY_HISTORY = db.Column(db.Integer, nullable=False)
    STRESS_IMMUNE = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(64), nullable=False)
    message = db.Column(db.String(256), nullable=False)
    risk_class = db.Column(db.String(32), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()


@app.route("/")
def index():
    # Home page — shows features and intro
    return render_template("home.html", features=FEATURES)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    submitted = False
    if request.method == "POST":
        submitted = True
    return render_template("contact.html", submitted=submitted)


@app.route("/predict", methods=["GET"])
def predict():
    return render_template("predict_main.html")


@app.route("/predict_heart", methods=["GET", "POST"])
def predict_heart():
    result = None
    form = {
        "AGE": "0",
        "GENDER": "0",
        "SMOKING": "0",
        "FINGER_DISCOLORATION": "0",
        "MENTAL_STRESS": "0",
        "EXPOSURE_TO_POLLUTION": "0",
        "LONG_TERM_ILLNESS": "0",
        "ENERGY_LEVEL": "0.0",
        "IMMUNE_WEAKNESS": "0",
        "BREATHING_ISSUE": "0",
        "ALCOHOL_CONSUMPTION": "0",
        "THROAT_DISCOMFORT": "0",
        "OXYGEN_SATURATION": "0.0",
        "CHEST_TIGHTNESS": "0",
        "FAMILY_HISTORY": "0",
        "SMOKING_FAMILY_HISTORY": "0",
        "STRESS_IMMUNE": "0"
    }

    if request.method == "POST":
        try:
            form.update({
                "AGE": request.form.get("AGE", "0"),
                "GENDER": request.form.get("GENDER", "0"),
                "SMOKING": request.form.get("SMOKING", "0"),
                "FINGER_DISCOLORATION": request.form.get("FINGER_DISCOLORATION", "0"),
                "MENTAL_STRESS": request.form.get("MENTAL_STRESS", "0"),
                "EXPOSURE_TO_POLLUTION": request.form.get("EXPOSURE_TO_POLLUTION", "0"),
                "LONG_TERM_ILLNESS": request.form.get("LONG_TERM_ILLNESS", "0"),
                "ENERGY_LEVEL": request.form.get("ENERGY_LEVEL", "0.0"),
                "IMMUNE_WEAKNESS": request.form.get("IMMUNE_WEAKNESS", "0"),
                "BREATHING_ISSUE": request.form.get("BREATHING_ISSUE", "0"),
                "ALCOHOL_CONSUMPTION": request.form.get("ALCOHOL_CONSUMPTION", "0"),
                "THROAT_DISCOMFORT": request.form.get("THROAT_DISCOMFORT", "0"),
                "OXYGEN_SATURATION": request.form.get("OXYGEN_SATURATION", "0.0"),
                "CHEST_TIGHTNESS": request.form.get("CHEST_TIGHTNESS", "0"),
                "FAMILY_HISTORY": request.form.get("FAMILY_HISTORY", "0"),
                "SMOKING_FAMILY_HISTORY": request.form.get("SMOKING_FAMILY_HISTORY", "0"),
                "STRESS_IMMUNE": request.form.get("STRESS_IMMUNE", "0")
            })
            
            result = predict_heart_disease(form)
            
            # Handle AJAX requests first
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify(result)
            
            # Persist prediction to database
            try:
                record = PredictionRecord(
                    user_id=current_user.id if current_user.is_authenticated else None,
                    AGE=parse_int(form["AGE"]),
                    GENDER=parse_int(form["GENDER"]),
                    SMOKING=parse_int(form["SMOKING"]),
                    FINGER_DISCOLORATION=parse_int(form["FINGER_DISCOLORATION"]),
                    MENTAL_STRESS=parse_int(form["MENTAL_STRESS"]),
                    EXPOSURE_TO_POLLUTION=parse_int(form["EXPOSURE_TO_POLLUTION"]),
                    LONG_TERM_ILLNESS=parse_int(form["LONG_TERM_ILLNESS"]),
                    ENERGY_LEVEL=parse_float(form["ENERGY_LEVEL"]),
                    IMMUNE_WEAKNESS=parse_int(form["IMMUNE_WEAKNESS"]),
                    BREATHING_ISSUE=parse_int(form["BREATHING_ISSUE"]),
                    ALCOHOL_CONSUMPTION=parse_int(form["ALCOHOL_CONSUMPTION"]),
                    THROAT_DISCOMFORT=parse_int(form["THROAT_DISCOMFORT"]),
                    OXYGEN_SATURATION=parse_float(form["OXYGEN_SATURATION"]),
                    FAMILY_HISTORY=parse_int(form["FAMILY_HISTORY"]),
                    SMOKING_FAMILY_HISTORY=parse_int(form["SMOKING_FAMILY_HISTORY"]),
                    STRESS_IMMUNE=parse_int(form["STRESS_IMMUNE"]),
                    prediction=result.get("label", "Unknown"),
                    message=result.get("message", ""),
                    risk_class=result.get("class", "risk-medium")
                )
                db.session.add(record)
                db.session.commit()
            except Exception as db_error:
                db.session.rollback()
                print(f"Database error saving heart prediction: {str(db_error)}")
        except Exception as e:
            result = {
                "label": "Error",
                "message": f"An error occurred: {str(e)}",
                "class": "risk-medium",
                "probability": {"risk": 50, "no_risk": 50}
            }
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify(result)

    return render_template(
        "predict_heart.html",
        result=result,
        form=form,
        instructions=INSTRUCTIONS,
        precautions=PRECAUTIONS
    )


@app.route("/cancer", methods=["GET", "POST"])
def cancer():
    cancer_result = None
    cancer_form = CANCER_FORM_DEFAULTS.copy()

    if request.method == "POST":
        try:
            cancer_form.update({
                feature_name: request.form.get(feature_name, CANCER_FORM_DEFAULTS[feature_name])
                for feature_name in CANCER_FEATURE_NAMES
            })
            cancer_result = predict_cancer(cancer_form)

            # Persist prediction to database
            try:
                record = CancerPredictionRecord(
                    user_id=current_user.id if current_user.is_authenticated else None,
                    GENDER=parse_cancer_feature_value("GENDER", cancer_form["GENDER"]),
                    AGE=parse_cancer_feature_value("AGE", cancer_form["AGE"]),
                    MENTAL_STRESS=parse_cancer_feature_value("MENTAL_STRESS", cancer_form["MENTAL_STRESS"]),
                    FINGER_DISCOLORATION=parse_cancer_feature_value("FINGER_DISCOLORATION", cancer_form["FINGER_DISCOLORATION"]),
                    SMOKING=parse_cancer_feature_value("SMOKING", cancer_form["SMOKING"]),
                    EXPOSURE_TO_POLLUTION=parse_cancer_feature_value("EXPOSURE_TO_POLLUTION", cancer_form["EXPOSURE_TO_POLLUTION"]),
                    ENERGYLEVEL=parse_cancer_feature_value("ENERGY_LEVEL", cancer_form["ENERGY_LEVEL"]),
                    IMMUNEWEAKNESS=parse_cancer_feature_value("IMMUNE_WEAKNESS", cancer_form["IMMUNE_WEAKNESS"]),
                    BREATHINGISSUE=parse_cancer_feature_value("BREATHING_ISSUE", cancer_form["BREATHING_ISSUE"]),
                    ALCOHOL_CONSUMPTION=parse_cancer_feature_value("ALCOHOL_CONSUMPTION", cancer_form["ALCOHOL_CONSUMPTION"]),
                    THROAT_DISCOMFORT=parse_cancer_feature_value("THROAT_DISCOMFORT", cancer_form["THROAT_DISCOMFORT"]),
                    OXYGEN_SATURATION=parse_cancer_feature_value("OXYGEN_SATURATION", cancer_form["OXYGEN_SATURATION"]),
                    CHEST_TIGHTNESS=parse_cancer_feature_value("CHEST_TIGHTNESS", cancer_form["CHEST_TIGHTNESS"]),
                    FAMILY_HISTORY=parse_cancer_feature_value("FAMILY_HISTORY", cancer_form["FAMILY_HISTORY"]),
                    SMOKING_FAMILY_HISTORY=parse_cancer_feature_value("SMOKING_FAMILY_HISTORY", cancer_form["SMOKING_FAMILY_HISTORY"]),
                    STRESS_IMMUNE=parse_cancer_feature_value("STRESS_IMMUNE", cancer_form["STRESS_IMMUNE"]),
                    prediction=cancer_result.get("label", "Unknown"),
                    message=cancer_result.get("message", ""),
                    risk_class=cancer_result.get("class", "risk-medium")
                )
                db.session.add(record)
                db.session.commit()
            except Exception as db_error:
                db.session.rollback()
                print(f"Database error saving cancer prediction: {str(db_error)}")

            # Handle AJAX requests after prediction and best-effort persistence.
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify(cancer_result)
        except Exception as e:
            cancer_result = {
                "label": "Error",
                "message": f"An error occurred: {str(e)}",
                "class": "risk-medium",
                "probability": {"risk": 50, "no_risk": 50}
            }
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify(cancer_result)

    return render_template(
        "cancer.html",
        cancer_result=cancer_result,
        cancer_form=cancer_form,
        cancer_fields=CANCER_FORM_FIELDS,
        instructions=CANCER_INSTRUCTIONS,
        precautions=CANCER_PRECAUTIONS
    )


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("profile"))


@app.route("/profile", methods=["GET", "POST"])
def profile():
    auth_message = None
    form = {
        "age": "0",
        "sex": "0",
        "height": "170",
        "weight": "70",
        "resting_bp": "120",
        "cholesterol": "180",
        "fasting_bs": "0",
        "exercise_minutes": "150",
        "sleep_hours": "7",
        "stress_level": "5"
    }
    dashboard = None
    predictions = []
    cancer_predictions = []

    if request.method == "POST":
        action = request.form.get("auth_action")

        if action == "register":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            if not email or not password or not confirm_password:
                auth_message = "Please fill in email and password fields."
            elif password != confirm_password:
                auth_message = "Passwords do not match."
            elif User.query.filter_by(email=email).first():
                auth_message = "An account with that email already exists."
            else:
                user = User(email=email, password_hash=generate_password_hash(password))
                db.session.add(user)
                db.session.commit()
                login_user(user)
                return redirect(url_for("profile"))

        elif action == "login":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                return redirect(url_for("profile"))
            auth_message = "Invalid email or password."

        elif action == "profile":
            if not current_user.is_authenticated:
                auth_message = "Please log in or sign up before saving your profile."
            else:
                form.update({
                    "age": request.form.get("age", "0"),
                    "sex": request.form.get("sex", "0"),
                    "height": request.form.get("height", "170"),
                    "weight": request.form.get("weight", "70"),
                    "resting_bp": request.form.get("resting_bp", "120"),
                    "cholesterol": request.form.get("cholesterol", "180"),
                    "fasting_bs": request.form.get("fasting_bs", "0"),
                    "exercise_minutes": request.form.get("exercise_minutes", "150"),
                    "sleep_hours": request.form.get("sleep_hours", "7"),
                    "stress_level": request.form.get("stress_level", "5")
                })
                profile_record = Profile(
                    user_id=current_user.id,
                    age=parse_int(form["age"]),
                    sex=parse_int(form["sex"]),
                    height=parse_int(form["height"]),
                    weight=parse_float(form["weight"]),
                    resting_bp=parse_int(form["resting_bp"]),
                    cholesterol=parse_int(form["cholesterol"]),
                    fasting_bs=parse_int(form["fasting_bs"]),
                    exercise_minutes=parse_int(form["exercise_minutes"]),
                    sleep_hours=parse_int(form["sleep_hours"]),
                    stress_level=parse_int(form["stress_level"])
                )
                db.session.add(profile_record)
                db.session.commit()
                dashboard = build_health_dashboard(form)

    if current_user.is_authenticated:
        latest_profile = Profile.query.filter_by(user_id=current_user.id).order_by(Profile.created_at.desc()).first()
        if latest_profile and dashboard is None:
            form = {
                "age": str(latest_profile.age),
                "sex": str(latest_profile.sex),
                "height": str(latest_profile.height),
                "weight": str(int(latest_profile.weight)),
                "resting_bp": str(latest_profile.resting_bp),
                "cholesterol": str(latest_profile.cholesterol),
                "fasting_bs": str(latest_profile.fasting_bs),
                "exercise_minutes": str(latest_profile.exercise_minutes),
                "sleep_hours": str(latest_profile.sleep_hours),
                "stress_level": str(latest_profile.stress_level)
            }
            dashboard = build_health_dashboard(form)

        predictions = PredictionRecord.query.filter_by(user_id=current_user.id).order_by(PredictionRecord.created_at.desc()).limit(10).all()
        cancer_predictions = CancerPredictionRecord.query.filter_by(user_id=current_user.id).order_by(CancerPredictionRecord.created_at.desc()).limit(10).all()

    return render_template("profile.html", form=form, dashboard=dashboard, predictions=predictions, cancer_predictions=cancer_predictions, auth_message=auth_message)


if __name__ == "__main__":
    app.run(debug=True)
