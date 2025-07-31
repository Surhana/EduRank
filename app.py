import streamlit as st
import pandas as pd
import numpy as np

# Function to normalize the decision matrix using vector normalization
def normalize_matrix(matrix):
    return matrix / np.linalg.norm(matrix, axis=0)

# Function to compute the weighted normalized matrix
def weighted_normalized_matrix(normalized_matrix, weights):
    return normalized_matrix * weights

# Function to compute PIS (Positive Ideal Solution) and NIS (Negative Ideal Solution)
def pis_nis(matrix, impacts):
    pis = []
    nis = []
    for i in range(matrix.shape[1]):
        if impacts[i] == '+':
            pis.append(np.max(matrix[:, i]))
            nis.append(np.min(matrix[:, i]))
        else:
            pis.append(np.min(matrix[:, i]))
            nis.append(np.max(matrix[:, i]))
    return np.array(pis), np.array(nis)

# Function to compute Euclidean distance from PIS and NIS
def euclidean_distance(matrix, pis, nis):
    pis_distance = np.sqrt(np.sum((matrix - pis) ** 2, axis=1))
    nis_distance = np.sqrt(np.sum((matrix - nis) ** 2, axis=1))
    return pis_distance, nis_distance

# Function to calculate the relative closeness to the ideal solution
def relative_closeness(pis_distance, nis_distance):
    return nis_distance / (pis_distance + nis_distance)

# Streamlit app setup
st.title("MAUT for Educational Innovation and Stock Ranking")
st.markdown("""
This app evaluates and ranks educational innovations using **Multi-Attribute Utility Theory (MAUT)** method.
It helps with social sustainability by selecting top innovations based on user-specified criteria.
""")

# File upload for decision matrix
uploaded_file = st.file_uploader("Upload Excel or CSV file with the decision matrix", type=["csv", "xlsx"])

# Fallback data if no file uploaded
def load_example():
    data = {
        'Alternative': ['Innovation A', 'Innovation B', 'Innovation C'],
        'Criterion 1': [7, 8, 6],
        'Criterion 2': [4, 5, 3],
        'Criterion 3': [9, 7, 8],
    }
    return pd.DataFrame(data)

# Load data
df = pd.read_csv(uploaded_file) if uploaded_file and uploaded_file.name.endswith("csv") else \
     pd.read_excel(uploaded_file) if uploaded_file else load_example()

# Display the dataframe
st.subheader("Decision Matrix")
st.write(df)

# Extract data
stocks = df.iloc[:, 0]  # The first column (Alternatives)
criteria = df.columns[1:]  # All columns except the first column (criteria)
data = df.iloc[:, 1:].astype(float)  # Convert the criteria columns to numeric values

# Input weights for each criterion
st.subheader("Input Weights (must sum to 1)")
weights = []
for i, col in enumerate(criteria):
    weight = st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0,
                             value=round(1/len(criteria), 3), step=0.001)
    weights.append(weight)

if round(sum(weights), 3) != 1.0:
    st.warning("⚠️ The weights must sum to 1. Please adjust.")

# Input impact for each criterion
st.subheader("Select Impact for Each Criterion")
impact = []
for i, col in enumerate(criteria):
    selected = st.radio(
        f"Is '{col}' a Benefit or Cost Criterion?",
        options=["Benefit (+)", "Cost (-)"],
        index=0,
        horizontal=True,
        key=f"impact_{i}"
    )
    impact.append("+" if "Benefit" in selected else "-")

# Step 1: Normalize the matrix using vector normalization
st.subheader("Step 1: Normalization")
normalized = normalize_matrix(data)

# Display normalized matrix
st.write(normalized)

# Step 2: Weighted Normalized Matrix
st.subheader("Step 2: Weighted Normalized Matrix")
weighted_matrix = weighted_normalized_matrix(normalized, weights)

# Display weighted matrix
st.write(weighted_matrix)

# Step 3: Calculate PIS and NIS
st.subheader("Step 3: Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS)")
pis, nis = pis_nis(weighted_matrix, impact)

# Display PIS and NIS
st.write("Positive Ideal Solution (PIS):", pis)
st.write("Negative Ideal Solution (NIS):", nis)

# Step 4: Euclidean distance from PIS and NIS
st.subheader("Step 4: Euclidean Distances from PIS and NIS")
pis_distance, nis_distance = euclidean_distance(weighted_matrix, pis, nis)

# Display Euclidean distances
st.write("Euclidean Distances from PIS:", pis_distance)
st.writ
