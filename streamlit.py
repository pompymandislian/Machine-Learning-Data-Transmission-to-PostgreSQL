import streamlit as st
import requests

# Streamlit UI
st.title('Welcome to Prediction')

st.subheader('Enter User Information')

transaction = st.number_input('transaction', min_value=0, max_value=100_000_000, step=10_000)
age = st.number_input('Age', min_value=0, max_value=120, step=1)
tenure = st.number_input('Tenure', min_value=0, step=1)
num_pages_visited = st.number_input('Number of Pages Visited', min_value=0, step=1)
has_credit_card = st.radio('Has Credit Card', [True, False])
items_in_cart = st.number_input('Items in Cart', min_value=0, step=1)

# Button to submit user information
if st.button('Submit'):
    # Collect user data
    user_data = {
        'transaction': transaction,
        "age": age,
        "tenure": tenure,
        "num_pages_visited": num_pages_visited,
        "has_credit_card": has_credit_card,
        "items_in_cart": items_in_cart
    }
    # Make POST request to FastAPI server
    try:
        response = requests.post("http://localhost:8000/predict/", json=user_data)
        response.raise_for_status()  # Raises an exception for 4xx or 5xx status codes
        result = response.json()
        prediction = result["purchase_prediction"]
        st.success(f"Purchase Prediction: {prediction}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to make prediction. Error: {e}")
