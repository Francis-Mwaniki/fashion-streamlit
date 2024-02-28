import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# Read the data from the CSV file
data = pd.read_csv('data.csv')

# Separate features and target variable
X = data[['Height', 'Gender', 'Color', 'Style']]
y = data['Item']

# One-hot encode categorical features
# encoder = OneHotEncoder(drop='first')
# One-hot encode categorical features
encoder = OneHotEncoder(drop='first')
X_encoded = pd.DataFrame(encoder.fit_transform(X[['Gender', 'Color', 'Style']]))
X_encoded.columns = encoder.get_feature_names_out(['Gender', 'Color', 'Style'])
X_encoded.reset_index(drop=True, inplace=True)
X = pd.concat([X[['Height']], X_encoded], axis=1)
# X_encoded = pd.DataFrame(encoder.fit_transform(X[['Gender', 'Color', 'Style']]), columns=encoder.get_feature_names_out(['Gender', 'Color', 'Style']))
# X = pd.concat([X[['Height']], X_encoded], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title('Fashion Item Recommender')

# User input
height_input = st.number_input('Height (in cm):', min_value=0.0)
gender_input = st.radio('Gender:', ['M', 'F'])
color_input = st.selectbox('Color:', data['Color'].unique())
style_input = st.radio('Style:', ['Casual', 'Formal', 'Sporty'])

# Make prediction
user_data = pd.DataFrame({
    'Height': [height_input],
    'Gender': [gender_input],
    'Color': [color_input],
    'Style': [style_input]
})

user_encoded = pd.DataFrame(encoder.transform(user_data[['Gender', 'Color', 'Style']]), columns=encoder.get_feature_names_out(['Gender', 'Color', 'Style']))
user_data = pd.concat([user_data[['Height']], user_encoded], axis=1)

predicted_item = model.predict(user_data)[0]

# Display result
st.subheader('Suggested Fashion Item:')
st.write(predicted_item)
