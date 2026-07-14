import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("hotel_booking_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="centered"
)

st.title("🏨 Hotel Booking Cancellation Prediction")

st.write("Enter the booking details below to predict whether the booking will be canceled.")

# -------------------------
# User Inputs
# -------------------------

number_of_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)

number_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

number_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)

number_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)

type_of_meal = st.selectbox(
    "Meal Plan",
    ["Meal Plan 1","Meal Plan 2","Meal Plan 3","Not Selected"]
)

car_parking_space = st.selectbox(
    "Car Parking",
    [0,1]
)

room_type = st.selectbox(
    "Room Type",
    ["Room_Type 1","Room_Type 2","Room_Type 3","Room_Type 4","Room_Type 5","Room_Type 6","Room_Type 7"]
)

lead_time = st.number_input(
    "Lead Time",
    min_value=0,
    value=50
)

market_segment = st.selectbox(
    "Market Segment",
    ["Online","Offline","Corporate","Complementary","Aviation"]
)

repeated = st.selectbox(
    "Repeated Guest",
    [0,1]
)

previous_cancel = st.number_input(
    "Previous Cancellations",
    min_value=0,
    value=0
)

previous_not_cancel = st.number_input(
    "Previous Non-Cancellations",
    min_value=0,
    value=0
)

average_price = st.number_input(
    "Average Room Price",
    min_value=0.0,
    value=100.0
)

special_requests = st.number_input(
    "Special Requests",
    min_value=0,
    value=1
)

reservation_month = st.slider(
    "Reservation Month",
    1,
    12,
    6
)

reservation_day = st.slider(
    "Reservation Weekday",
    0,
    6,
    3
)

total_guests = number_of_adults + number_of_children

total_nights = number_of_weekend_nights + number_of_week_nights

has_children = 1 if number_of_children > 0 else 0

is_long_stay = 1 if total_nights >= 5 else 0

# -------------------------
# Prediction
# -------------------------

if st.button("Predict"):

    input_df = pd.DataFrame({

        "number of adults":[number_of_adults],
        "number of children":[number_of_children],
        "number of weekend nights":[number_of_weekend_nights],
        "number of week nights":[number_of_week_nights],
        "type of meal":[type_of_meal],
        "car parking space":[car_parking_space],
        "room type":[room_type],
        "lead time":[lead_time],
        "market segment type":[market_segment],
        "repeated":[repeated],
        "P-C":[previous_cancel],
        "P-not-C":[previous_not_cancel],
        "average price":[average_price],
        "special requests":[special_requests],
        "total_guests":[total_guests],
        "total_nights":[total_nights],
        "has_children":[has_children],
        "is_long_stay":[is_long_stay],
        "reservation_month":[reservation_month],
        "reservation_day":[reservation_day]

    })

    processed = preprocessor.transform(input_df)

    processed = scaler.transform(processed)

    prediction = model.predict(processed)[0]

    if prediction == 1:
        st.error("Prediction: Booking is likely to be Cancelled")
    else:
        st.success("Prediction: Booking is likely to be Not Cancelled")