import os
import pickle
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for
from auth import init_auth, login_manager, UserMixin, login_user, logout_user, current_user
from database import db, init_db
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask("Medicare")
app.secret_key = "Medicare-ai@123#"

basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(basedir, 'medicare.db')}"
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
    "Provide truthful demographic and lifestyle details for the cancer risk prediction.",
    "Use the selection fields for encoded values like gender, smoker status, and income level.",
    "Submit the form to receive an estimate based on the trained cancer model.",
    "Review results carefully and seek medical advice rather than relying on this prediction alone."
]

CANCER_PRECAUTIONS = [
    "A predictive model is not a diagnostic tool; consult a healthcare professional for any concerns.",
    "Lifestyle factors like smoking, employment, and social habits are included for model estimation.",
    "Keep follow-up appointments and screenings as recommended by your doctor."
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
        parse_int(form.get("age", 0)),
        parse_int(form.get("sex", 0)),
        parse_int(form.get("chest_pain", 0)),
        parse_int(form.get("resting_bp", 0)),
        parse_int(form.get("cholesterol", 0)),
        parse_int(form.get("fasting_bs", 0)),
        parse_int(form.get("resting_ecg", 0)),
        parse_int(form.get("max_hr", 0)),
        parse_int(form.get("exercise_angina", 0)),
        parse_float(form.get("oldpeak", 0.0)),
        parse_int(form.get("st_slope", 0)),
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


CANCER_ENCODING = {
    "Gender": {"Female": 0, "Male": 1},
    "Smoking": {"No": 0, "Yes": 1},
    "GeneticRisk": {"Low": 0, "Medium": 1, "High": 2},
    "CancerHistory": {"No": 0, "Yes": 1}
}


def encode_cancer_value(value, mapping):
    if value in mapping:
        return mapping[value]
    try:
        numeric = int(value)
        if numeric in mapping.values():
            return numeric
    except (TypeError, ValueError):
        pass
    return 0


def predict_cancer(form):
    payload = [
        encode_cancer_value(form.get("gender", "Female"), CANCER_ENCODING["Gender"]),
        parse_int(form.get("age", 0)),
        float(form.get("bmi", 0.0)),
        encode_cancer_value(form.get("smoking", "No"), CANCER_ENCODING["Smoking"]),
        encode_cancer_value(form.get("genetic_risk", "Medium"), CANCER_ENCODING["GeneticRisk"]),
        float(form.get("physical_activity", 0.0)),
        float(form.get("alcohol_intake", 0.0)),
        encode_cancer_value(form.get("cancer_history", "No"), CANCER_ENCODING["CancerHistory"])
    ]

    if cancer_model is not None:
        prediction = cancer_model.predict([payload])
        try:
            prob = cancer_model.predict_proba([payload])[0]
            risk_prob = int(prob[1] * 100) if len(prob) > 1 else 50
            no_risk_prob = 100 - risk_prob
        except:
            risk_prob = 50
            no_risk_prob = 50
        
        pred_label = str(prediction[0])
        if pred_label.lower() in ["yes", "1", "true"]:
            return {
                "label": "Cancer Risk Detected",
                "message": "The model predicts a possible cancer risk. Please consult a qualified healthcare professional.",
                "class": "risk-high",
                "probability": {"risk": risk_prob, "no_risk": no_risk_prob}
            }
        return {
            "label": "No Cancer Detected",
            "message": "The model predicts a low cancer risk based on the provided inputs.",
            "class": "risk-low",
            "probability": {"risk": risk_prob, "no_risk": no_risk_prob}
        }

    return {
        "label": "Cancer model unavailable",
        "message": "No cancer model file was found. Place cancer_best_model.pkl in the project root to enable cancer risk prediction.",
        "class": "risk-medium"
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
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)
    chest_pain = db.Column(db.Integer, nullable=False)
    resting_bp = db.Column(db.Integer, nullable=False)
    cholesterol = db.Column(db.Integer, nullable=False)
    fasting_bs = db.Column(db.Integer, nullable=False)
    resting_ecg = db.Column(db.Integer, nullable=False)
    max_hr = db.Column(db.Integer, nullable=False)
    exercise_angina = db.Column(db.Integer, nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    st_slope = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(64), nullable=False)
    message = db.Column(db.String(256), nullable=False)
    risk_class = db.Column(db.String(32), nullable=False)


class CancerPredictionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    gender = db.Column(db.Integer, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    smoking = db.Column(db.Integer, nullable=False)
    genetic_risk = db.Column(db.Integer, nullable=False)
    physical_activity = db.Column(db.Float, nullable=False)
    alcohol_intake = db.Column(db.Float, nullable=False)
    cancer_history = db.Column(db.Integer, nullable=False)
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
        "age": "0",
        "sex": "0",
        "chest_pain": "0",
        "resting_bp": "120",
        "cholesterol": "180",
        "fasting_bs": "0",
        "resting_ecg": "0",
        "max_hr": "130",
        "exercise_angina": "0",
        "oldpeak": "1.0",
        "st_slope": "0"
    }

    if request.method == "POST":
        form.update({
            "age": request.form.get("age", "0"),
            "sex": request.form.get("sex", "0"),
            "chest_pain": request.form.get("chest_pain", "0"),
            "resting_bp": request.form.get("resting_bp", "120"),
            "cholesterol": request.form.get("cholesterol", "180"),
            "fasting_bs": request.form.get("fasting_bs", "0"),
            "resting_ecg": request.form.get("resting_ecg", "0"),
            "max_hr": request.form.get("max_hr", "130"),
            "exercise_angina": request.form.get("exercise_angina", "0"),
            "oldpeak": request.form.get("oldpeak", "1.0"),
            "st_slope": request.form.get("st_slope", "0")
        })
        result = predict_heart_disease(form)
        record = PredictionRecord(
            user_id=current_user.id if current_user.is_authenticated else None,
            age=parse_int(form["age"]),
            sex=parse_int(form["sex"]),
            chest_pain=parse_int(form["chest_pain"]),
            resting_bp=parse_int(form["resting_bp"]),
            cholesterol=parse_int(form["cholesterol"]),
            fasting_bs=parse_int(form["fasting_bs"]),
            resting_ecg=parse_int(form["resting_ecg"]),
            max_hr=parse_int(form["max_hr"]),
            exercise_angina=parse_int(form["exercise_angina"]),
            oldpeak=parse_float(form["oldpeak"]),
            st_slope=parse_int(form["st_slope"]),
            prediction=result["label"],
            message=result["message"],
            risk_class=result["class"]
        )
        db.session.add(record)
        db.session.commit()

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
    cancer_form = {
        "gender": "Female",
        "age": "0",
        "bmi": "25.0",
        "smoking": "No",
        "genetic_risk": "Medium",
        "physical_activity": "5.0",
        "alcohol_intake": "2.0",
        "cancer_history": "No"
    }

    if request.method == "POST":
        cancer_form.update({
            "gender": request.form.get("gender", "Female"),
            "age": request.form.get("age", "0"),
            "bmi": request.form.get("bmi", "25.0"),
            "smoking": request.form.get("smoking", "No"),
            "genetic_risk": request.form.get("genetic_risk", "Medium"),
            "physical_activity": request.form.get("physical_activity", "5.0"),
            "alcohol_intake": request.form.get("alcohol_intake", "2.0"),
            "cancer_history": request.form.get("cancer_history", "No")
        })
        cancer_result = predict_cancer(cancer_form)
        record = CancerPredictionRecord(
            user_id=current_user.id if current_user.is_authenticated else None,
            gender=encode_cancer_value(cancer_form["gender"], CANCER_ENCODING["Gender"]),
            age=parse_int(cancer_form["age"]),
            bmi=parse_float(cancer_form["bmi"]),
            smoking=encode_cancer_value(cancer_form["smoking"], CANCER_ENCODING["Smoking"]),
            genetic_risk=encode_cancer_value(cancer_form["genetic_risk"], CANCER_ENCODING["GeneticRisk"]),
            physical_activity=parse_float(cancer_form["physical_activity"]),
            alcohol_intake=parse_float(cancer_form["alcohol_intake"]),
            cancer_history=encode_cancer_value(cancer_form["cancer_history"], CANCER_ENCODING["CancerHistory"]),
            prediction=cancer_result["label"],
            message=cancer_result["message"],
            risk_class=cancer_result["class"]
        )
        db.session.add(record)
        db.session.commit()

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify(cancer_result)

    return render_template(
        "cancer.html",
        cancer_result=cancer_result,
        cancer_form=cancer_form,
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
