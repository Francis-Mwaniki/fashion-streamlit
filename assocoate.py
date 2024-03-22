import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

def predict_associated_item(gender_input, color_input, style_input):
    # Read data and remove duplicates
    data = pd.read_csv("associate.csv").drop_duplicates(subset=['Color', 'Style', 'Gender', 'Rating']).reset_index(drop=True)

    # Filter data based on gender and style
    filtered_data = data[(data['Gender'] == gender_input) & (data['Style'] == style_input)]
    

    # Filter out unrealistic associations during training
    realistic_data = data[data['Association'] != 'Blouse'] if gender_input == 'M' else data

    # Impute missing values in categorical columns (replace with appropriate strategies)
    imputer = SimpleImputer(strategy='most_frequent')  # Use 'most_frequent' for categorical columns
    data_imputed = imputer.fit_transform(realistic_data[['Gender', 'Color', 'Style']])
    data_imputed = pd.DataFrame(data_imputed, columns=['Gender', 'Color', 'Style'])  # Convert back to DataFrame

    # Encode categorical variables
    encoder = OneHotEncoder(drop='first',handle_unknown='ignore')
    encoded_data = encoder.fit_transform(data_imputed[['Gender', 'Color', 'Style']])

    # Concatenate encoded data with original columns
    X = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Gender', 'Color', 'Style']))
    y = realistic_data['Association']  # Target variable is still 'Item'

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Prepare user input for prediction
    user_data = pd.DataFrame({
        'Gender': [gender_input],
        'Color': [color_input],
        'Style': [style_input]
    })

    # Handle unknown categories for user input
    user_data_imputed = imputer.transform(user_data)
    user_encoded = encoder.transform(user_data_imputed)

    # Convert the sparse matrix to a DataFrame
    user_encoded_df = pd.DataFrame(user_encoded.toarray(), columns=encoder.get_feature_names_out(['Gender', 'Color', 'Style']))

    # Make prediction
    predicted_associated_item = model.predict(user_encoded_df)[0]

    return predicted_associated_item
