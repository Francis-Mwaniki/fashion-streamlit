import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import streamlit as st

# Read data and remove duplicates
data = pd.read_csv("associate.csv").drop_duplicates(subset=['Color', 'Style', 'Gender', 'Rating']).reset_index(drop=True)

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# Impute missing values in categorical columns (replace with appropriate strategies)
imputer = SimpleImputer(strategy='most_frequent')  # Use 'most_frequent' for categorical columns
data_imputed = imputer.fit_transform(data[['Gender', 'Color', 'Style']])
data_imputed = pd.DataFrame(data_imputed, columns=['Gender', 'Color', 'Style'])  # Convert back to DataFrame

# Encode categorical variables
encoder = OneHotEncoder(drop='first')
encoded_data = encoder.fit_transform(data_imputed)

# Concatenate encoded data with original columns
X = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Gender', 'Color', 'Style']))
y = data['Association']  # Target variable is still 'Item'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title='Fashion', page_icon=':shirt:', layout='centered', initial_sidebar_state='expanded')
st.title('Fashion Advisor')

with st.sidebar:
    gender_input = st.selectbox("Gender:", ["M", "F"])
    color_input = st.text_input("Color:")
    style_input = st.selectbox("Style:", ["Casual", "Formal", "Sporty"])

if st.button("Predict Associated Fashion Item"):
    try:
        # Prepare user input for prediction
        user_data = pd.DataFrame({
            'Gender': [gender_input],
            'Color': [color_input],
            'Style': [style_input]
        })

        # One-hot encode categorical features
        user_encoded = encoder.transform(user_data)

        # Convert the sparse matrix to a DataFrame
        user_encoded_df = pd.DataFrame(user_encoded.toarray(), columns=encoder.get_feature_names_out(['Gender', 'Color', 'Style']))

        # Make prediction
        predicted_associated_item = model.predict(user_encoded_df)[0]

        # Container for the output
        with st.container():
            st.success(f" {predicted_associated_item} would be a great choice as an associated item!")
            # Provide additional advice based on the predicted item
            # You can customize this advice based on your domain knowledge
            additional_advice = "Consider pairing it with your chosen item to complete your outfit."
            complete_advice = f"{predicted_associated_item} is associated with your choice. {additional_advice}"
            st.write(complete_advice)

    except ValueError as e:
        st.error(str(e))
