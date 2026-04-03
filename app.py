import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("best_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="E-Commerce Profit Margin Predictor", layout="wide")

st.title("🛒 E-Commerce Profit Margin Predictor")
st.write("Predict Profit Margin using order and customer details")

# ---------------------------
# Sidebar Inputs
# ---------------------------

st.sidebar.header("Input Features")

price = st.sidebar.number_input("Price", min_value=0.0, value=100.0)
discount = st.sidebar.number_input("Discount (%)", min_value=0.0, value=10.0)
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
delivery_time_days = st.sidebar.number_input("Delivery Time (Days)", min_value=1, value=5)
shipping_cost = st.sidebar.number_input("Shipping Cost", min_value=0.0, value=5.0)
customer_age = st.sidebar.number_input("Customer Age", min_value=10, max_value=100, value=30)

year = st.sidebar.selectbox("Year", [2022, 2023, 2024, 2025])
month = st.sidebar.selectbox("Month", list(range(1, 13)))
day = st.sidebar.selectbox("Day", list(range(1, 32)))
quarter = st.sidebar.selectbox("Quarter", [1, 2, 3, 4])

category = st.sidebar.selectbox(
    "Category",
    ["Beauty", "Electronics", "Fashion", "Grocery", "Home", "Sports", "Toys"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["COD", "Credit Card", "Debit Card", "PayPal", "UPI", "Wallet"]
)

region = st.sidebar.selectbox(
    "Region",
    ["Central", "East", "North", "South", "West"]
)

returned = st.sidebar.selectbox("Returned", ["No", "Yes"])

customer_gender = st.sidebar.selectbox(
    "Customer Gender",
    ["Female", "Male", "Other"]
)

# ---------------------------
# Feature Engineering
# ---------------------------

total_amount = price * quantity * (1 - discount / 100)

input_dict = {
    'price': price,
    'discount': discount,
    'quantity': quantity,
    'delivery_time_days': delivery_time_days,
    'total_amount': total_amount,
    'shipping_cost': shipping_cost,
    'customer_age': customer_age,
    'year': year,
    'month': month,
    'day': day,
    'quarter': quarter,
}

columns = [
    'category_Beauty','category_Electronics','category_Fashion',
    'category_Grocery','category_Home','category_Sports','category_Toys',
    'payment_method_COD','payment_method_Credit Card',
    'payment_method_Debit Card','payment_method_PayPal',
    'payment_method_UPI','payment_method_Wallet',
    'region_Central','region_East','region_North','region_South','region_West',
    'returned_No','returned_Yes',
    'customer_gender_Female','customer_gender_Male','customer_gender_Other'
]

for col in columns:
    input_dict[col] = 0

input_dict[f'category_{category}'] = 1
input_dict[f'payment_method_{payment_method}'] = 1
input_dict[f'region_{region}'] = 1
input_dict[f'returned_{returned}'] = 1
input_dict[f'customer_gender_{customer_gender}'] = 1

input_df = pd.DataFrame([input_dict])

feature_order = [
    'price', 'discount', 'quantity', 'delivery_time_days', 'total_amount',
    'shipping_cost', 'customer_age', 'year', 'month', 'day', 'quarter',
    'category_Beauty', 'category_Electronics', 'category_Fashion',
    'category_Grocery', 'category_Home', 'category_Sports', 'category_Toys',
    'payment_method_COD', 'payment_method_Credit Card',
    'payment_method_Debit Card', 'payment_method_PayPal',
    'payment_method_UPI', 'payment_method_Wallet', 'region_Central',
    'region_East', 'region_North', 'region_South', 'region_West',
    'returned_No', 'returned_Yes', 'customer_gender_Female',
    'customer_gender_Male', 'customer_gender_Other'
]

input_df = input_df[feature_order]

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Profit Margin"):

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    st.success(f"📈 Predicted Profit Margin: {prediction:.4f}")

    st.subheader("Input Features")
    st.dataframe(input_df)
