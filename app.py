import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="💖 Love Prediction App", page_icon="❤️", layout="wide")
st.title("💖 Love Prediction App - Emoji Style 💖")
st.write("Slide the emojis to express your feelings for each behavior:")


@st.cache_resource
def train_fresh_model():
    try:
        # Load Data
        df = pd.read_csv('psychology_love_prediction_dataset.csv')
        
        features = [
            'communication_frequency', 'emotional_support', 'initiates_conversation', 
            'remembers_small_details', 'smiles_often', 'respects_boundaries', 
            'gets_jealous', 'spends_time'
        ]
        target = 'love'
        
     
        max_vals = df[features].max()
        min_vals = df[features].min()
        mean_vals = df[features].mean()

   
        X = df[features]
        y = df[target]
        
 
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

       
        model = LogisticRegression()
        model.fit(X_scaled, y)

        return model, scaler, max_vals, min_vals, mean_vals

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# Load the logic
model, scaler, max_vals, min_vals, mean_vals = train_fresh_model()

if model:
   
    def emoji_slider(label, col_name):
        options = ["No 💔", "Neutral 😐", "Yes ❤️"]
        choice = st.select_slider(label, options=options, value="Neutral 😐")

        if choice == "No 💔":
            return min_vals[col_name] 
        elif choice == "Neutral 😐":
            return mean_vals[col_name] 
        else:
            return max_vals[col_name] 

    st.markdown("### Select your response for each feature:")

    # Layout columns
    col1, col2 = st.columns(2)


    with col1:
        comm = emoji_slider("💬 Communication Frequency", 'communication_frequency')
        emo = emoji_slider("🤝 Emotional Support", 'emotional_support')
        init = emoji_slider("📞 Initiates Conversation", 'initiates_conversation')
        rem = emoji_slider("📝 Remembers Small Details", 'remembers_small_details')

    with col2:
        smile = emoji_slider("😊 Smiles Often", 'smiles_often')
        bound = emoji_slider("🚧 Respects Boundaries", 'respects_boundaries')
        jeal = emoji_slider("😠 Gets Jealous", 'gets_jealous')
        time = emoji_slider("⏰ Spends Time Together", 'spends_time')


    if st.button("Predict Your Love ❤️", use_container_width=True):
   
        input_data = pd.DataFrame({
            'communication_frequency': [comm], 
            'emotional_support': [emo], 
            'initiates_conversation': [init], 
            'remembers_small_details': [rem], 
            'smiles_often': [smile], 
            'respects_boundaries': [bound], 
            'gets_jealous': [jeal], 
            'spends_time': [time]
        })

        # 2. Scale the input (Crucial Step!)
        input_scaled = scaler.transform(input_data)
        
        # 3. Predict Probability
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        
        # 4. Result
        if probability > 0.5:
            st.success(f"Prediction: ❤️ Love! (Confidence: {probability:.2f})")
            st.balloons()
        else:
            st.error(f"Prediction: 💔 No Love (Confidence: {probability:.2f})")
else:
    st.warning("Could not load data. Please check 'psychology_love_prediction_dataset.csv'")