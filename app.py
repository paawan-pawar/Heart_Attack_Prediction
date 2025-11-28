# app.py
# Streamlit Heart Attack Predictor (local-only: no upload widget)
# Edit DEFAULT_MODEL_PATH and DEFAULT_DATA_PATH at the top before running.

import streamlit as st
import pandas as pd
import os
import joblib
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Optional

st.set_page_config(page_title="Heart Attack Predictor", layout="wide")

# ------------------ EDIT THESE PATHS ------------------
# Use a Windows full path or a relative path. Example:
# DEFAULT_MODEL_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\Heart_attack\model.joblib"
# DEFAULT_DATA_PATH  = r"C:\Users\LENOVO\OneDrive\Desktop\Heart_attack\heart_data.csv"
DEFAULT_MODEL_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\Heart_attack\rfc.jb"   # <--- change if needed
DEFAULT_DATA_PATH  = r"C:\Users\LENOVO\OneDrive\Desktop\Heart_attack\Heart_attack.csv"     # <--- change if needed
# ----------------------------------------------------

FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "Diabetes", "PhysActivity", "HvyAlcoholConsump", "MentHlth", "PhysHlth",
    "Sex", "Age"
]

POSITIVE_CLASS_LABEL = 1  # label that indicates heart-attack in your model

st.title("Heart Attack Predictor")

col1, col2 = st.columns([2, 1])
with col2:
    st.write("**Mention Sex/YES/NO as:**")
    st.code(f"Female/NO: 0 \nMale/YES: 1", language="bash")


def try_load_model(path: str):
    try:
        if os.path.exists(path):
            mdl = joblib.load(path)
            return mdl, None
        else:
            return None, f"Model file not found at: {path}"
    except Exception as exc:
        return None, f"Failed to load model: {exc}\n{traceback.format_exc()}"

model, model_load_error = try_load_model(DEFAULT_MODEL_PATH)

if model is not None:
    st.success(f"Loaded model--Default")
    try:
        st.sidebar.subheader("Model info")
        st.sidebar.write(type(model).__name__)
        st.sidebar.write("Accuracy:89%")
        if hasattr(model, "n_features_in_"):
            st.sidebar.write(f"n_features_in_: {model.n_features_in_}")
        if hasattr(model, "classes_"):
            st.sidebar.write(f"Classes: {list(model.classes_)}")
    except Exception:
        pass
else:
    st.error(model_load_error or "No model loaded.")

# Option to train from local CSV if model missing
if model is None:
    st.markdown("---")
    st.header("Train model from local CSV (optional)")
    st.write(
        "If you don't have the model file available, train a RandomForestClassifier from a local CSV. "
        "The CSV must contain the feature columns listed below and a 'target' column (1 = heart-attack, 0 = no)."
    )
    st.write("Required features:")
    st.write(FEATURES)
    if st.button("Train model from DEFAULT_DATA_PATH now"):
        if not os.path.exists(DEFAULT_DATA_PATH):
            st.error(f"Training CSV not found at: {DEFAULT_DATA_PATH}")
        else:
            try:
                df = pd.read_csv(DEFAULT_DATA_PATH)
                missing = [c for c in FEATURES + ["target"] if c not in df.columns]
                if missing:
                    st.error(f"CSV is missing required columns: {missing}")
                else:
                    st.info("Preparing data and training. This might take some time.")
                    X = df[FEATURES]
                    y = df["target"]
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_val)
                    acc = accuracy_score(y_val, preds)
                    st.success(f"Training complete — validation accuracy: {acc:.4f}")
                    try:
                        joblib.dump(clf, DEFAULT_MODEL_PATH)
                        st.success(f"Model saved to {DEFAULT_MODEL_PATH}")
                        model = clf
                    except Exception as e:
                        st.error(f"Failed to save model: {e}")
            except Exception as e:
                st.error(f"Training failed: {e}\n{traceback.format_exc()}")

if model is None:
    st.warning("No model available. Edit DEFAULT_MODEL_PATH or train from DEFAULT_DATA_PATH.")
    st.stop()

# Single-row prediction UI
st.markdown("---")
st.header("Predict — Single Input")

def widget_for_feature(name: str):
    binary_like = {"HighBP","HighChol","CholCheck","Smoker","Stroke","Diabetes","PhysActivity","HvyAlcoholConsump","Sex"}
    integer_like = {"MentHlth","PhysHlth","Age","Education","Income"}
    if name in binary_like:
        return st.selectbox(name, options=[0,1], index=0, help=f"Binary input for {name} (0 or 1)")
    if name in integer_like:
        return st.number_input(name, value=0, step=1, format="%d")
    return st.number_input(name, value=25.0, format="%.3f")

inputs = {}
cols = st.columns(3)
for i, f in enumerate(FEATURES):
    with cols[i % 3]:
        inputs[f] = widget_for_feature(f)

input_df = pd.DataFrame([inputs])
st.write("### Input preview")
st.dataframe(input_df)

if st.button("Predict chance of heart attack"):
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0]
            # locate index for positive class (1)
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                pos_idx = classes.index(POSITIVE_CLASS_LABEL) if POSITIVE_CLASS_LABEL in classes else 1
            else:
                pos_idx = 1
            chance = float(prob[pos_idx])
        else:
            pred = model.predict(input_df)[0]
            chance = 1.0 if pred == POSITIVE_CLASS_LABEL else 0.0

        st.metric("Chance of heart attack", f"{chance*100:.2f}%")
        label = "YES" if chance >= 0.5 else "NO"
        st.write(f"**Prediction:** {label}")

        if hasattr(model, "predict_proba"):
            # nicely show prob per class
            classes = list(model.classes_) if hasattr(model, "classes_") else [0,1]
            st.write(pd.DataFrame([prob], columns=[f"prob_class_{c}" for c in classes]))
    except Exception as e:
        st.error(f"Prediction failed: {e}\n{traceback.format_exc()}")

# Batch predictions from DEFAULT_DATA_PATH (no upload)
st.markdown("---")
st.header("Batch predictions from DEFAULT_DATA_PATH (no upload)")
st.write("If a CSV exists at DEFAULT_DATA_PATH and contains the required feature columns, run batch predictions and download results.")

if os.path.exists(DEFAULT_DATA_PATH):
    try:
        df_all = pd.read_csv(DEFAULT_DATA_PATH)
        st.write("Preview of data:")
        st.dataframe(df_all.head())
        if st.button("Run batch predictions on DEFAULT_DATA_PATH"):
            missing = [c for c in FEATURES if c not in df_all.columns]
            if missing:
                st.error(f"CSV missing columns: {missing}")
            else:
                X_all = df_all[FEATURES]
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_all)
                    if hasattr(model, "classes_"):
                        classes = list(model.classes_)
                        pos_idx = classes.index(POSITIVE_CLASS_LABEL) if POSITIVE_CLASS_LABEL in classes else 1
                    else:
                        pos_idx = 1
                    df_all["heart_attack_chance"] = probs[:, pos_idx]
                else:
                    df_all["heart_attack_chance"] = model.predict(X_all)
                df_all["heart_attack_label"] = df_all["heart_attack_chance"].apply(lambda x: "YES" if x >= 0.5 else "NO")
                st.write("### Predictions")
                st.dataframe(df_all.head(100))
                csv_bytes = df_all.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to read DEFAULT_DATA_PATH: {e}\n{traceback.format_exc()}")
else:
    st.info(f"No CSV found at DEFAULT_DATA_PATH: {DEFAULT_DATA_PATH}. Place a CSV there to enable batch predictions.")

