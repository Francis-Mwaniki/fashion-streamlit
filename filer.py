import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer  # Import library for imputation
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# Set your OpenAI API key
client = OpenAI(
    api_key= os.environ.get('OPENAI_API_KEY')
)

# Read data
data = pd.read_csv('data.csv')

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# Impute missing values in numerical columns (replace with appropriate strategies)
imputer = SimpleImputer(strategy='mean')  # Use 'mean' for numerical columns
data_numeric = imputer.fit_transform(data[['Height']])  # Impute only numerical columns
data_numeric = pd.DataFrame(data_numeric, columns=['Height'])  # Convert back to DataFrame

print(data_numeric)

# Impute missing values in categorical columns (replace with appropriate strategies)
imputer = SimpleImputer(strategy='most_frequent')  # Use 'most_frequent' for categorical columns
data_categorical = imputer.fit_transform(data[['Height', 'Gender', 'Color', 'Style']])
data_categorical = pd.DataFrame(data_categorical, columns=[' Height', 'Gender', 'Color', 'Style'])  # Convert back to DataFrame

print(data_categorical)
# Combine numerical and categorical data
data_imputed = pd.concat([data_numeric, data_categorical], axis=1)

print(data_imputed)
encoder = OneHotEncoder(drop='first')
encoded_data = encoder.fit_transform(data_imputed[['Height', 'Gender', 'Color', 'Style']])

print(encoded_data)

# Concatenate with the original "Height" column
X = pd.concat([data_imputed[['Height']], pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Height','Gender', 'Color', 'Style']))], axis=1)
print(f'X: {X}')
print(f'data[Item]: {data["Item"]}')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['Item'], test_size=0.2, random_state=42)


# Streamlit UI
ui_css = '''
<style>
.stApp > header {
    background-color: transparent;
}

.stApp {
    margin: auto;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    overflow: auto; 
    background-color: transparent;
    animation: gradient 15s ease infinite;
    background-size: 400% 400%;
    background-attachment: fixed;
}
   
   
    </style>
'''
st.set_page_config(page_title='Fashion', page_icon=':shirt:', layout='centered', initial_sidebar_state='expanded')
st.title('Fashion')
st.markdown(ui_css, unsafe_allow_html=True)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#Generate advices based on user input
def generate_autocompletion(prompt):
    fashion_role = f"""
    You are a fashion advisor. 
    Advice the user on fashion item selected and their selected preferences.
    add some uniqueness to the advice.
    let the user know why the fashion item is the best choice for them. using the user's preferences.
"""

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": fashion_role}, {"role": "user", "content": prompt}],
    )
    # Extract the output text from the API response
    output_text = response.choices[0].message.content
    print(f'Output text: {response.choices[0].message.content}')
    return st.markdown(output_text)

# Function to suggest the best fashion item based on user inputs
def suggest_fashion_item(height, gender, color, style):
    height_lower_bound = 150
    height_upper_bound = 200
    if height < height_lower_bound or height > height_upper_bound:
        raise ValueError(f'Height should be between {height_lower_bound} and {height_upper_bound} cm')

    # Prepare user input for prediction
    user_data = pd.DataFrame({
        'Height':  [height],
        'Gender': [gender],
        'Color': [color],
        'Style': [style]
    })

    # One-hot encode categorical features
    user_encoded = encoder.transform(user_data[['Height', 'Gender', 'Color', 'Style']])

    # Convert the sparse matrix to a DataFrame
    user_encoded_df = pd.DataFrame(user_encoded.toarray(), columns=encoder.get_feature_names_out(['Height', 'Gender', 'Color', 'Style']))

    # Concatenate user_data and user_encoded_df
    user_data_encoded = pd.concat([user_data[['Height']], user_encoded_df], axis=1)

    print(f'user_data_encoded: {user_data_encoded}')

    # Make prediction
    predicted_item = model.predict(user_data_encoded)[0]
    print(f'predicted_item: {predicted_item}')

    return predicted_item

with st.sidebar:
    height_input = st.slider("Height (in cm):", 150.0, 200.0, 175.0)
    gender_input = st.selectbox("Gender:", ["M", "F"])
    color_input = st.text_input("Color:")
    style_input = st.selectbox("Style:", ["Casual", "Formal", "Sporty"])
    defaultHeight = 175.0

# predicted_item = suggest_fashion_item(height_input,gender_input,color_input,style_input)
if st.button("Predict Fashion Item"):
    try:
        predicted_item = suggest_fashion_item(defaultHeight, gender_input, color_input, style_input)
        # Container for the output
        with st.container():
            st.success(f" {predicted_item} would be a great choice for you!")
            #parse the  defaultHeight, gender_input, color_input, style_input plus the predicted_item to the OpenAI API
            with st.spinner('crafing the best advice for you...'):
                prompt = f"Preferences: {height_input,gender_input,color_input,style_input} Use predicted item as the fashion item without recommending other: {predicted_item}"
                generate_autocompletion(prompt)
            
    except ValueError as e:
        st.error(str(e))
        st.stop()
