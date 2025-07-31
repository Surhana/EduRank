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
st.title("MAUT Method for Ranking Alternatives")
st.markdown("""
    This app evaluates and ranks alternatives using the **Multi-Attribute Utility Theory (MAUT)** method.
    Users can upload a decision matrix and specify weights and impacts for criteria.
""")

# File upload for decision matrix
uploaded_file = st.file_uploader("Upload an Excel or CSV file with the decision matrix", type=["csv", "xlsx"])

# Default example dataset if no file uploaded
if uploaded_file is None:
    st.warning("No file uploaded, using example dataset")
    data = {
        'Alternative': ['A1', 'A2', 'A3'],
        'Criterion 1': [8, 7, 9],
        'Criterion 2': [3, 2, 4],
        'Criterion 3': [6, 8, 7],
    }
    df = pd.DataFrame(data)
else:
    # Load the user-uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

# Display the dataframe
st.subheader("Decision Matrix")
st.write(df)

# Input for weights and impacts
weights_input = st.text_input("Enter weights for each criterion (comma separated, e.g., 0.3, 0.2, 0.5)", "0.3, 0.2, 0.5")
weights = np.array([float(x) for x in weights_input.split(',')])

impacts_input = st.text_input("Enter impacts for each criterion (comma separated, + for benefit, - for cost)", "+, +, -")
impacts = impacts_input.split(',')

# Normalize the decision matrix
matrix = df.drop(columns=['Alternative']).to_numpy()
normalized_matrix = normalize_matrix(matrix)

# Compute the weighted normalized matrix
weighted_matrix = weighted_normalized_matrix(normalized_matrix, weights)

# Compute PIS and NIS
pis, nis = pis_nis(weighted_matrix, impacts)

# Compute Euclidean distances
pis_distance, nis_distance = euclidean_distance(weighted_matrix, pis, nis)

# Compute relative closeness
closeness = relative_closeness(pis_distance, nis_distance)

# Add closeness to dataframe
df['Closeness'] = closeness

# Rank alternatives based on closeness
df['Rank'] = df['Closeness'].rank(ascending=False)

# Display results
st.subheader("Results")
st.write(df)

# Highlight the top-ranked alternative
st.markdown(f"**Top Ranked Alternative: {df['Alternative'].iloc[0]}**")

# Option to download the results as CSV
st.download_button(
    label="Download Results as CSV",
    data=df.to_csv(index=False),
    file_name="ranking_results.csv",
    mime="text/csv"
)

# Example of how to run: streamlit run topsis_app.py
