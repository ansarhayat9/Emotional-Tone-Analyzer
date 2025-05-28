import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from PIL import Image
import base64

# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    return joblib.load("model/text_emotion.pkl")

pipe_lr = load_model()

# --- Emotion Config ---
emotions_emoji_dict = {
    "anger": "😠", 
    "disgust": "🤮", 
    "fear": "😨", 
    "happy": "😊", 
    "joy": "😂", 
    "neutral": "😐", 
    "sad": "😢",
    "sadness": "😔", 
    "shame": "😳", 
    "surprise": "😲"
}

# --- Gradient Color Mapping for Emotions ---
emotion_gradients = {
    "anger": "linear-gradient(135deg, #ff4d4d, #cc0000)",
    "disgust": "linear-gradient(135deg, #66ff66, #009900)",
    "fear": "linear-gradient(135deg, #6666ff, #0000cc)",
    "happy": "linear-gradient(135deg, #ffff66, #ffcc00)",
    "joy": "linear-gradient(135deg, #ff9933, #ff6600)",
    "neutral": "linear-gradient(135deg, #d9d9d9, #999999)",
    "sad": "linear-gradient(135deg, #66b3ff, #0066cc)",
    "sadness": "linear-gradient(135deg, #99ccff, #0066cc)",
    "shame": "linear-gradient(135deg, #cc99ff, #9933cc)",
    "surprise": "linear-gradient(135deg, #ff66b3, #cc0066)"
}

# --- Emotion Insights (Dynamic Responses) ---
emotion_insights = {
    "anger": "This text contains strong language suggesting frustration or anger. Taking deep breaths can help calm emotions.",
    "disgust": "The language indicates repulsion or dislike. Consider reframing thoughts for a more neutral perspective.",
    "fear": "This text expresses anxiety or fear. If this is about a real concern, reaching out for support may help.",
    "happy": "The tone is positive and uplifting! Spread the joy! 😊",
    "joy": "This text is full of happiness and laughter! Keep the good vibes going! 🎉",
    "neutral": "The tone is balanced and factual. No strong emotions detected.",
    "sad": "The language suggests sadness. If you're feeling down, talking to someone can help.",
    "sadness": "The text conveys melancholy. Remember, it's okay to seek support when needed.",
    "shame": "The wording indicates discomfort or embarrassment. Self-compassion can help ease these feelings.",
    "surprise": "This text expresses astonishment or shock! Something unexpected is happening here!"
}

# --- Prediction Functions ---
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# --- Custom CSS & Background ---
def set_bg_hack():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                background-attachment: fixed;
                background-size: cover;
            }}
            .st-emotion-card {{
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: transform 0.3s, box-shadow 0.3s;
                margin-bottom: 20px;
            }}
            .st-emotion-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            }}
            .stTextArea textarea {{
                border-radius: 10px !important;
                padding: 15px !important;
                font-size: 16px;
            }}
            .stButton button {{
                background: linear-gradient(135deg, #6e8efb, #a777e3);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s;
            }}
            .stButton button:hover {{
                transform: scale(1.02);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .highlight {{
                background: linear-gradient(90deg, #ffcc00, #ff9900);
                padding: 2px 8px;
                border-radius: 5px;
                color: white;
                font-weight: bold;
            }}
            .header {{
                font-size: 2.5em;
                color: #4a4a4a;
                text-align: center;
                margin-bottom: 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Main App ---
def main():
    set_bg_hack()  # Apply custom background
    
    # --- Header with Gradient (Replacement for colored_header) ---
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #014c43, #10c1ac); 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center;
                    margin-bottom: 30px;">
            <h1 style="color: white; margin: 0;">🎭 Emotion Detection AI</h1>
            <p style="color: white; opacity: 0.9;">Discover the emotions behind any text with AI-powered analysis.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # --- Sidebar (App Info) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1998/1998664.png", width=80)
        st.markdown("### **About This App**")
        st.markdown("""
        This AI analyzes text to detect emotions like happiness, sadness, anger, and more.
        - **How it works**: Enter text → AI predicts emotion → See results!
        - **Uses**: Sentiment analysis, mental health awareness, customer feedback.
        """)
        
        st.markdown("---")
        st.markdown("### **Detected Emotions**")
        for emotion, emoji in emotions_emoji_dict.items():
            st.markdown(f"{emoji} **{emotion.capitalize()}**")
        
        st.markdown("---")
        st.markdown("Built with ❤️ using **Streamlit** & **Scikit-learn**")
        st.markdown("Ansar Hayat❤️")

    # --- Main Content ---
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.form("emotion_form"):
            raw_text = st.text_area(
                "**Enter your text here:**", 
                placeholder="Example: 'I feel so happy today! The weather is amazing.'",
                height=200
            )
            submitted = st.form_submit_button("**Analyze Emotions** 🚀")
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4476/4476876.png", width=150)
        st.caption("_AI-powered emotion detection_")

    if submitted and raw_text:
        with st.spinner("🔍 Analyzing emotions..."):
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            confidence = np.max(probability)
            
            # --- Emotion Card (Animated) ---
            st.markdown(
                f"""
                <div class="st-emotion-card" style="border-left: 5px solid; border-image: {emotion_gradients[prediction]}; border-image-slice: 1;">
                    <h2 style="margin-bottom: 0;">{prediction.capitalize()} {emotions_emoji_dict[prediction]}</h2>
                    <p style="font-size: 18px;">Confidence: <span class="highlight">{confidence:.0%}</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # --- Emotion Probability Chart ---
            st.subheader("📊 **Emotion Distribution**")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotion", "probability"]
            
            chart = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X("emotion:N", sort="-y", title="Emotion"),
                y=alt.Y("probability:Q", title="Probability", axis=alt.Axis(format=".0%")),
                color=alt.Color("emotion:N", scale=alt.Scale(
                    domain=list(emotions_emoji_dict.keys()),
                    range=["#ff4d4d", "#66ff66", "#6666ff", "#ffff66", "#ff9933", "#d9d9d9", "#66b3ff", "#99ccff", "#cc99ff", "#ff66b3"]
                )),
                tooltip=["emotion", "probability"]
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
            
            # --- Emotion Insights ---
            st.subheader("💡 **What This Means**")
            st.info(emotion_insights[prediction])
            
            # --- Text Preview ---
            with st.expander("📝 **View Your Text**"):
                st.write(raw_text)

if __name__ == "__main__":
    main()