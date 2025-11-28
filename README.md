Machine Learning Model
📌 Heart Attack Prediction App (Streamlit + Machine Learning)

This project is a Streamlit-based interactive web app that predicts the chance of a heart attack using a trained machine learning model.
It supports:

✔️ Local model loading (no file upload)

✔️ Downloading large .joblib model from a GitHub Release or cloud storage

✔️ YES/NO prediction + Probability (%)

✔️ Batch prediction from a local CSV

✔️ Docker deployment

✔️ Streamlit Cloud deployment (model downloaded on startup)

🔧 Features Used in the Model

The trained model uses these 15 features:

HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
Diabetes
PhysActivity
HvyAlcoholConsump
MentHlth
PhysHlth
Sex
Age
Education
Income

✔️ YES / NO Encoding
Feature	0	1
HighBP	No	Yes
HighChol	No	Yes
CholCheck	No	Yes
Smoker	No	Yes
Stroke	No	Yes
Diabetes	No	Yes
PhysActivity	No	Yes
HvyAlcoholConsump	No	Yes
✔️ Sex Column Encoding
Value	Meaning
0	Female
1	Male

(If your dataset uses different encoding, update this table.)

🚀 Running Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the app
python -m streamlit run app.py

3️⃣ Model download

The app uses a MODEL_URL to download the .joblib file automatically if it's not found locally.

Set environment variable (Windows PowerShell):

$env:MODEL_URL="https://github.com/<your-user>/<your-repo>/releases/download/v1.0/rfc_model.joblib"


Linux/macOS:

export MODEL_URL="https://github.com/<your-user>/<your-repo>/releases/download/v1.0/rfc_model.joblib"


If the release is private, also add:

export GITHUB_TOKEN="your_personal_access_token"

📁 Project Structure
Heart_attack_app/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── entrypoint.sh
├── README.md
└── models/                # Auto-created (ignored in git)


.gitignore should contain:

/models/*
*.joblib

🐳 Running with Docker
Build image
docker build -t heart-app .

Run container (auto downloads model)
docker run -p 8501:8501 -e MODEL_URL="https://github.com/<owner>/<repo>/releases/download/v1.0/rfc_model.joblib" heart-app

Run with local model file
docker run -p 8501:8501 -v $(pwd)/models:/app/models heart-app

☁️ Deploying to Streamlit Cloud

Streamlit Cloud cannot accept Docker, but it can download your model on startup.

Push your repo to GitHub

Go to https://share.streamlit.io

Create a new app connected to your repo

Add environment variable in App Settings → Secrets:

MODEL_URL = "https://github.com/<owner>/<repo>/releases/download/v1.0/rfc_model.joblib"


If your release is private:

GITHUB_TOKEN = "your_personal_access_token"


Streamlit Cloud will download the model each time the app starts.

🧠 Model Prediction Output

The app shows:

Chance of heart attack (%)

Prediction: YES / NO

(If available) Class probabilities

📦 Dataset Information (If included)

If you include dataset details:

Sex: 0 = Female, 1 = Male

Binary features: 0 = No, 1 = Yes

BMI, Age, Education, Income are numeric

MentHlth and PhysHlth range from 0–30 days

(Modify according to your dataset.)

🛠 Technologies Used

Streamlit

Scikit-learn

Python 3.11

Docker

GitHub Releases for model hosting

❤️ Credits

Created by Paawan Pawar
If you use this project, please ⭐ star the repository!