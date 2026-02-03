import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import gzip
import os


#knn=joblib.load("knn_pca_model.pkl")
#nb = joblib.load("naive_bayes_model.pkl")
#dt = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
svm=joblib.load("svm_model.pkl")
#lr=joblib.load("logistic_regression_model.pkl")
#rf=joblib.load("rf_model.pkl")
# ---------------- FEATURE EXTRACTION ----------------
def extract_features(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features.reshape(1, -1)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection")
st.write("Upload a facial image to check driver alertness using ML models.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = cv2.imdecode(
        np.frombuffer(uploaded_file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    features = extract_features(image)
    features = scaler.transform(features)

    ##pred_knn = knn.predict(features)[0]
    #pred_nb = nb.predict(features)[0]
    #pred_dt = dt.predict(features)[0]
    pred_svm=svm.predict(features)[0]
    #pred_lr=lr.predict(features)[0]
    #pred_rf=rf.predict(features)[0]
    label = {
        1: "DROWSY 😴",
        0: "NON-DROWSY 🙂"
    }

    st.subheader("🔍 Model Prediction")
    #st.write(f"**KNN:** {label[pred_knn]}")
    #st.write(f"**Naive Bayes:** {label[pred_nb]}")
    #st.write(f"**Decision Tree:** {label[pred_dt]}")
    st.write(f"**-->** {label[pred_svm]}")
    #st.write(f"**Logistic regression:** {label[pred_lr]}")
    #st.write(f"**random forest:** {label[pred_rf]}")