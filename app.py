import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import streamlit as st
from dotenv import load_dotenv
from assocoate import predict_associated_item

load_dotenv()

# Read data remove duplicates Item, Color, Style, Gender, Rating
data = pd.read_csv("data2.csv").drop_duplicates(subset=['Item', 'Color', 'Style','Gender','Rating']).reset_index(drop=True)

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
y = data['Item']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

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
st.title('Fashion Advisor')
st.markdown(ui_css, unsafe_allow_html=True)

with st.sidebar:
    gender_input = st.selectbox("Gender:", ["M", "F"])
    color_input = st.text_input("Color:")
    style_input = st.selectbox("Style:", ["Casual", "Formal", "Sporty"])

if st.button("Predict Fashion Item"):
    try:
        # Prepare user input for prediction
        user_data = pd.DataFrame({
            'Gender': [gender_input],
            'Color': [color_input],
            'Style': [style_input]
        })
        results = predict_associated_item(gender_input=gender_input, color_input=color_input, style_input=style_input)

        # One-hot encode categorical features
        user_encoded = encoder.transform(user_data)

        # Convert the sparse matrix to a DataFrame
        user_encoded_df = pd.DataFrame(user_encoded.toarray(), columns=encoder.get_feature_names_out(['Gender', 'Color', 'Style']))

        # Make prediction
        predicted_item = model.predict(user_encoded_df)[0]

        # Container for the output
        with st.container():
            # detailed advisory  based on the predicted item and results(associative item)
            # st.success(f" You can consider {predicted_item} and {results} for your outfit")
            # Define additional advice based on parameters
            if style_input == "Casual":
                if gender_input == "M":
                    additional_advice = "For a relaxed and laid-back vibe, consider comfortable yet stylish options that reflect your unique personality. Experiment with versatile pieces that transition seamlessly from day to night."
                elif gender_input == "F":
                    additional_advice = "For a chic and effortless ensemble, incorporate trendy and comfortable pieces into your wardrobe. Layering and accessorizing can add depth to your look, expressing your individual style with confidence."
            elif style_input == "Sporty":
                if gender_input == "M":
                    additional_advice = "Opt for performance-driven pieces that offer style and functionality. Look for breathable fabrics and athletic silhouettes that keep you comfortable during active pursuits."
                elif gender_input == "F":
                    additional_advice = "Explore activewear options that blend fashion and function. Prioritize comfort and mobility while expressing your personal style with confidence."
            elif style_input == "Formal":
                if gender_input == "M":
                    additional_advice = "Embrace classic elegance with tailored pieces that exude sophistication. Invest in timeless wardrobe staples such as well-fitted suits and polished dress shoes for a sharp and distinguished look."
                elif gender_input == "F":
                    additional_advice = "Make a statement with chic and polished ensembles that command attention. Choose tailored separates and elegant accessories to elevate your formal look with grace and confidence."

            # Define original advice based on gender and style
            if gender_input == "M":
                if style_input == "Casual":
                    advice = f"For a casual look, consider pairing a {results} with a"
                elif style_input == "Sporty":
                    advice = f"For a sporty look, consider incorporating a {results} with a"
                elif style_input == "Formal":
                    advice = f"For a formal look, consider adding a touch of elegance with a {results} with a"
            elif gender_input == "F":
                if style_input == "Casual":
                    advice = f"For a casual look, consider pairing a {results} with a"
                elif style_input == "Sporty":
                    advice = f"For a sporty look, consider incorporating a {results} with a"
                elif style_input == "Formal":
                    advice = f"For a formal look, consider adding a touch of elegance with a {results} with a"

            # Concatenate the advice with predicted item for an attractive suggestion
            complete_advice = f"{advice} {predicted_item}. {additional_advice} Showcase your unique style and make a lasting impression wherever you go."

            # Display the complete advice
            st.markdown(f"**Fashion Advice:**")
            st.info(complete_advice)

    except ValueError as e:
        st.error(str(e))
