import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "C:/Users/gargi/MoviePrediction/movie_metadata_renamed.csv"
data = pd.read_csv(file_path)

# Handle missing values
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Define features and target variable
features = ['budget', 'duration', 'num_critic_for_reviews', 'imdb_score']
data['Category'] = data['imdb_score'].apply(lambda score: 'Hit' if score >= 6 else ('Average' if score >= 3 else 'Flop'))
label_encoder = LabelEncoder()
data['Category_encoded'] = label_encoder.fit_transform(data['Category'])

X = data[features]
y = data['Category_encoded']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Streamlit user interface
st.title("Movie Success Predictor")
st.write("Enter movie details:")

# Create input fields for the selected features
budget = st.number_input("Budget", min_value=0)
duration = st.number_input("Duration (minutes)", min_value=0)
num_critic_for_reviews = st.number_input("Number of Critics Reviews", min_value=0)
imdb_score = st.number_input("IMDb Score", min_value=0.0, max_value=10.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[budget, duration, num_critic_for_reviews, imdb_score]])
    prediction = rf.predict(input_data)
    category = label_encoder.inverse_transform(prediction)[0]
    st.write(f"The predicted category is: **{category}**")
