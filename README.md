# 🎭 Text Emotion Detection App

This is a simple AI-powered web app that detects emotions from user-input text. It was built as a **beginner project** to learn the basics of **Machine Learning**, **Natural Language Processing (NLP)**, and **Streamlit**.

---

## 💡 Purpose

I created this project as part of my learning journey in **Artificial Intelligence**. The goal was to understand how text data can be processed and classified into different emotions using machine learning models.

---

## 🧠 How It Works

1. **Text Input**: Users enter any text.
2. **Preprocessing**: The text is cleaned (user handles removed, stopwords filtered).
3. **Prediction**: A trained machine learning model (Logistic Regression) predicts the emotion.
4. **Results**:
   - The predicted emotion with an emoji and confidence score
   - A bar chart showing the probability distribution across all emotions
   - An insight message describing the detected emotion

---

## 🔧 Tools & Technologies

- Python
- Pandas & NumPy
- Scikit-learn
- NeatText
- Streamlit
- Joblib (for saving/loading models)
- Altair (for visualization)

---

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/text-emotion-detector.git
   cd text-emotion-detector
Install dependencies:
   pip install -r requirements.txt

streamlit run app.py


    
