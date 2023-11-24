import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Function to make predictions
def predict_efficiency(temperature, pressure, time, humidity, pH):
    # Create a NumPy array from the user inputs
    input_data = np.array([[temperature, pressure, time, humidity, pH]])

    # Use the pre-trained model to make predictions
    efficiency_prediction = xgb_model.predict(input_data)[0]

    return efficiency_prediction

# Streamlit UI
st.set_page_config(page_title="SustainAI - Raw Material Management", page_icon="🌿")

# Custom CSS for blinking headline and description box
custom_styles = """
<style>
@keyframes blink {
    50% {
        opacity: 0;
    }
}
.blinking-text {
    font-size: 24px;
    color: green;
    animation: blink 1s infinite;
}

.shadow-box {
    background-color: #f0f0f0;
    padding: 20px;
    box-shadow: 5px 5px 10px #888888;
    border-radius: 10px;
}

.bold-italic {
    font-weight: bold;
    font-style: italic;
}
</style>
"""

st.markdown(custom_styles, unsafe_allow_html=True)

# Headline
st.markdown('<p class="blinking-text">SustainAI Developed by Aadil Gani</p>', unsafe_allow_html=True)

# Small Description inside a shadow box
with st.markdown('<div class="shadow-box">', unsafe_allow_html=True):
    st.markdown('<p class="bold-italic">Raw material management plays a crucial role in our daily lives, '
                'from the production of goods to environmental sustainability. '
                'Traditional methods often fall short, impacting both efficiency and the environment. '
                'This is where SustainAI steps in, bringing the power of smart technology to revolutionize '
                'how we manage raw materials.</p>', unsafe_allow_html=True)

# User inputs
temperature = st.slider('Temperature', min_value=0, max_value=200, value=100)
pressure = st.slider('Pressure', min_value=0, max_value=100, value=50)
time = st.slider('Time', min_value=0, max_value=24, value=12)
humidity = st.slider('Humidity', min_value=0, max_value=100, value=50)
pH = st.slider('pH', min_value=0, max_value=14, value=7)

# Predict button
if st.button('Predict Efficiency'):
    efficiency_result = predict_efficiency(temperature, pressure, time, humidity, pH)

    # Display the prediction result
    st.write(f'Predicted Efficiency: {efficiency_result:.2f}')

    # Feature importance plot
    feature_names = ['Temperature', 'Pressure', 'Time', 'Humidity', 'pH']
    feature_importance = xgb_model.feature_importances_

    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_names, ax=ax, palette="viridis")
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    st.pyplot(fig)
