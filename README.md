# 📘 AI Sign Language Recognition

🔗 **Dataset & Model Download (Google Drive)**  
https://drive.google.com/drive/folders/1Nk7gm2Dc64tySBxxcGfeXNrbpqnq_-cn?usp=drive_link

---

## 🖐️ Project Overview

AI Sign Language Recognition is a real-time web application that detects and classifies hand gestures into alphabet letters using Computer Vision and Machine Learning.

This project helps:
- Beginners learn sign language
- Perform real-time gesture recognition
- Demonstrate AI-based hand tracking
- Build assistive communication tools

---

## 🚀 How the App Works

1. User opens the Streamlit web app.
2. User clicks **Start Camera**.
3. Camera captures live video frames.
4. MediaPipe detects 21 hand landmarks.
5. 63 features (x, y, z coordinates) are extracted.
6. Trained Machine Learning model predicts the alphabet.
7. Detected symbol is displayed in the UI.

---

## 🧠 System Architecture

```
Camera Frame
     ↓
MediaPipe Hand Detection
     ↓
Feature Extraction (x, y, z coordinates)
     ↓
Machine Learning Model Prediction
     ↓
Streamlit UI Display
```

---

## 💡 Features

- Real-time hand gesture detection
- Fast and interactive UI
- Dark/Light theme toggle
- Clean and attractive dashboard
- Alphabet recognition (A–Z)
- Modular project structure

---

## 📁 Project Structure

```
sign-language-gesture-recognition/
│
├── frontend/
│   └── app.py
│
├── feature_extraction/
│   └── extract_features.py
│
├── model/
│   └── model.pkl
│
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧾 Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Rikhil777/sign-language-gesture-recognition.git
cd sign-language-gesture-recognition
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download Dataset & Model

Download from:

https://drive.google.com/drive/folders/1Nk7gm2Dc64tySBxxcGfeXNrbpqnq_-cn?usp=drive_link

Download and place inside project root:

```
dataset/
features.csv
model/model.pkl
```

⚠ Do NOT push these large files to GitHub.

---

## ▶️ Run the Application

From project root:

```bash
streamlit run frontend/app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🛠 Train the Model (Optional)

If you want to retrain the model:

```bash
python train_model.py
```

This will:
- Load features.csv
- Train classifier
- Save model.pkl

---

## 🔍 Feature Extraction (Optional)

If using new dataset:

```bash
python feature_extraction/extract_features.py
```

This will:
- Read images from dataset folder
- Extract hand landmarks
- Generate features.csv

---

## 💻 Tech Stack

- OpenCV – Camera processing
- MediaPipe – Hand landmark detection
- Scikit-Learn – Machine learning classifier
- Streamlit – Web frontend
- NumPy – Numerical processing

---

## 📌 Important Notes

Do NOT commit:

```
venv/
dataset/
features.csv
model/model.pkl
```

Your `.gitignore` should include:

```
__pycache__/
*.pyc
venv/
dataset/
features.csv
model/model.pkl
.vscode/
.DS_Store
Thumbs.db
```

---

## 🧠 One-Line Explanation

The system extracts 21 hand landmark coordinates using MediaPipe, converts them into 63 numerical features, and uses a trained machine learning classifier to predict the corresponding alphabet in real-time.

---

## 👥 Team Instructions

- Install dependencies using requirements.txt
- Download dataset and model from Drive link
- Run Streamlit app locally
- Do NOT push large files to GitHub

---

## 🙌 Final Output

✔ Real-time alphabet recognition  
✔ Clean UI with theme toggle  
✔ Fast detection pipeline  
✔ Proper ML workflow  

---
## 📷 Screenshots
<img width="1920" height="1200" alt="Screenshot (147)" src="https://github.com/user-attachments/assets/ff2b7b3a-d370-4beb-a81d-a0ac39bc833c" />
<br>
<br>
<img width="1920" height="1200" alt="Screenshot (148)" src="https://github.com/user-attachments/assets/0a11726a-17c1-4623-a9a3-c1cfdf41d844" />


---

🚀 Built with Computer Vision + Machine Learning
