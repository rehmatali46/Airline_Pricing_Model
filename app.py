import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import gdown

MODEL_URL = 'https://drive.google.com/uc?id=https://drive.google.com/file/d/1ZQMNvwM4I-fiWE2eMy8RWCRsdbs87qGx/view?usp=drive_link'
MODEL_PATH = 'dynamic_pricing_model.pkl'

if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model file...'):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# --- Load model and features
model = joblib.load('dynamic_pricing_model.pkl')
features = joblib.load('model_features.pkl')

# --- Load dataset for visualization
df = pd.read_csv('indian_airlines_dynamic_pricing_large.csv')

st.set_page_config(page_title="Dynamic Pricing Strategy", layout="wide")
st.title("✈️ Dynamic Pricing Recommendation Engine (Airlines)")

# --- User Input Section
st.header("🔢 Enter Flight and Booking Details")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        booking_window_days = st.slider('Booking Window (days before flight)', 1, 60, 15)
        route_popularity = st.slider('Route Popularity (0.5 - 1.0)', 0.5, 1.0, 0.75)
        base_fare = st.number_input('Base Fare (₹)', 1000, 20000, 5000)
    with col2:
        seats_remaining = st.slider('Seats Remaining', 1, 100, 30)
        competitor_avg_fare = st.number_input('Competitor Average Fare (₹)', 1000, 20000, 5200)
        is_weekend = st.selectbox('Is Weekend?', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    with col3:
        demand_level = st.selectbox('Demand Level', ['Low', 'Medium', 'High'])
        season = st.selectbox('Season', ['Peak', 'Off-Peak', 'Festival', 'Holiday'])
        seat_class = st.selectbox('Seat Class', ['Economy', 'Business', 'Premium Economy'])
        user_segment = st.selectbox('User Segment', ['Business', 'Leisure', 'Student', 'Family'])
    submitted = st.form_submit_button("Recommend Price")

# --- Prepare input for prediction
input_dict = {
    'booking_window_days': booking_window_days,
    'route_popularity': route_popularity,
    'base_fare': base_fare,
    'seats_remaining': seats_remaining,
    'competitor_avg_fare': competitor_avg_fare,
    'is_weekend': is_weekend,
}
# One-hot encoding for categorical variables
for cat, options in {
    'demand_level': ['Low', 'Medium', 'High'],
    'season': ['Peak', 'Off-Peak', 'Festival', 'Holiday'],
    'seat_class': ['Economy', 'Business', 'Premium Economy'],
    'user_segment': ['Business', 'Leisure', 'Student', 'Family']
}.items():
    for opt in options:
        col_name = f"{cat}_{opt}"
        input_dict[col_name] = 1 if eval(cat) == opt else 0
for feat in features:
    if feat not in input_dict:
        input_dict[feat] = 0
input_df = pd.DataFrame([input_dict])[features]

# --- Prediction Result
if submitted:
    pred = model.predict(input_df)[0]
    st.success(f"💡 Recommended Dynamic Fare: ₹{pred:.2f}")

# --- Data Visualizations Section
st.header("📊 Data Visualizations & Insights")

# 1. Distribution of Dynamic Fare
st.subheader("Distribution of Dynamic Fare")
fig1 = px.histogram(df, x='dynamic_fare', nbins=50, title='Dynamic Fare Distribution')
st.plotly_chart(fig1, use_container_width=True)

# 2. Dynamic Fare by Demand Level (Box Plot)
st.subheader("Dynamic Fare by Demand Level")
fig2 = px.box(df, x='demand_level', y='dynamic_fare', color='demand_level',
              title='Dynamic Fare by Demand Level')
st.plotly_chart(fig2, use_container_width=True)

# 3. Average Dynamic Fare by Season (Bar Chart)
st.subheader("Average Dynamic Fare by Season")
season_avg = df.groupby('season')['dynamic_fare'].mean().reset_index()
fig3 = px.bar(season_avg, x='season', y='dynamic_fare', color='season',
              title='Average Dynamic Fare by Season')
st.plotly_chart(fig3, use_container_width=True)

# 4. Dynamic Fare vs. Competitor Average Fare (Scatter Plot)
st.subheader("Dynamic Fare vs. Competitor Average Fare")
fig4 = px.scatter(df, x='competitor_avg_fare', y='dynamic_fare', color='demand_level',
                  title='Dynamic Fare vs. Competitor Avg Fare', trendline='ols')
st.plotly_chart(fig4, use_container_width=True)

# 5. Dynamic Fare by User Segment (Bar Chart)
st.subheader("Average Dynamic Fare by User Segment")
user_seg_avg = df.groupby('user_segment')['dynamic_fare'].mean().reset_index()
fig5 = px.bar(user_seg_avg, x='user_segment', y='dynamic_fare', color='user_segment',
              title='Average Dynamic Fare by User Segment')
st.plotly_chart(fig5, use_container_width=True)

# --- Optional: Show raw data
with st.expander("Show raw data"):
    st.dataframe(df.head(100))
