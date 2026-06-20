import streamlit as st
import pandas as pd
import joblib

# Background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #080616;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = joblib.load("c_segmentation.pkl")

# Title
st.title("🛍️ Customer Segmentation App")
st.write("Enter customer details to predict customer segment.")

# User Input

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

income = st.number_input(
    "Annual Income (k$)",
    min_value=1,
    max_value=200,
    value=50
)

spending_score = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)


# Cluster names
cluster_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "Young High Spenders",
    3: "Wealthy Conservative Customers",
    4: "Budget Customers"
}


# Short cluster information
cluster_info = {
    0: "Medium income & medium spending. Regular customer base.",
    1: "High income & high spending. Most valuable customers.",
    2: "Young customers with low income but very high spending.",
    3: "High income but low spending. Spend carefully.",
    4: "Low income & low spending. Price-sensitive customers."
}


# Prediction

if st.button("Predict Segment"):

    input_data = pd.DataFrame({
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending_score]
    })


    cluster = model.predict(input_data)[0]

    segment = cluster_names.get(cluster, "Unknown Segment")

    st.success(f"Customer Segment: {segment}")

    st.info(f"Cluster Number: {cluster}")

    st.subheader("📌 Customer Insight")
    st.write(cluster_info[cluster])


st.write("Total Clusters:", model.n_clusters)