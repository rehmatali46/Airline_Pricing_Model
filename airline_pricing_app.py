import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("airline_dynamic_pricing.csv")
    data['booking_date'] = pd.to_datetime(data['booking_date'])
    data['departure_date'] = pd.to_datetime(data['departure_date'])
    data['lead_time_days'] = (data['departure_date'] - data['booking_date']).dt.days
    features = ['occupancy_rate', 'lead_time_days', 'route_distance', 'base_price', 'competitor_price', 'fuel_surcharge', 'route_popularity']
    X = data[features]
    y = data['dynamic_price']
    return X, y, data

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_test, y_test

# Predict price
def predict_price(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    return model.predict(input_scaled)[0]

# Streamlit app with new design
st.set_page_config(page_title="Airline Dynamic Pricing", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a2a44;
        color: #d3e0ea;
    }
    .stButton>button {
        background-color: #4a7ab4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stSlider, .stNumberInput {
        background-color: #2e4057;
        color: #d3e0ea;
    }
    .stHeader {
        background-color: #4a7ab4;
        padding: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Navigation bar
st.markdown('<div class="stHeader"><h2>Airline Pricing Control Panel</h2></div>', unsafe_allow_html=True)

# Collapsible input panel
with st.expander("Adjust Flight Parameters", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        occupancy = st.slider("Occupancy Rate (%)", 0.0, 1.0, 0.8, 0.01)
        lead_time = st.slider("Lead Time (days)", 0, 120, 60)
        distance = st.slider("Route Distance (km)", 400, 2000, 1500, 100)
    with col2:
        base_price = st.number_input("Base Price (₹)", 3000, 6000, 5000)
        competitor_price = st.number_input("Competitor Price (₹)", 3000, 6000, 4500)
        fuel_surcharge = st.number_input("Fuel Surcharge (₹)", 200, 600, 400)
    customer_segment = st.selectbox("Customer Segment", ["Leisure", "Business", "Group"])
    route_pop = st.slider("Route Popularity", 0.5, 1.0, 0.8, 0.05)

# Map customer segment to a dummy variable (simplified encoding)
segment_map = {"Leisure": 0, "Business": 1, "Group": 2}
segment_value = segment_map[customer_segment]

# Input data (adjust features to exclude is_peak_season)
input_data = [occupancy, lead_time, distance, base_price, competitor_price, fuel_surcharge, route_pop]


X, y, df = load_data()
model, scaler, X_test, y_test = train_model(X, y)

# Prediction
predicted_price = predict_price(model, scaler, input_data)
st.subheader("Predicted Dynamic Price: ₹{:.2f}".format(predicted_price))

# Visualization
st.subheader("Price Comparison")
fig = go.Figure(data=[
    go.Bar(name='Base Price', x=['Price'], y=[base_price]),
    go.Bar(name='Competitor Price', x=['Price'], y=[competitor_price]),
    go.Bar(name='Predicted Price', x=['Price'], y=[predicted_price])
])
fig.update_layout(barmode='group', template="plotly_dark", title="Price Analysis", xaxis_title="Category", yaxis_title="Price (₹)", plot_bgcolor='#2e4057', paper_bgcolor='#2e4057', font_color="#d3e0ea")
st.plotly_chart(fig)

# Stats
st.subheader("Model Performance")
test_pred = model.predict(scaler.transform(X_test))
r2 = model.score(scaler.transform(X_test), y_test)
st.write(f"R² Score: {r2:.2f}")