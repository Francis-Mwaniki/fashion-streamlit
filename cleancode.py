import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import sklearn.preprocessing
from sklearn.impute import SimpleImputer  # Import library for imputation


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
# # Create DataFrame based on the encoded output
# if encoded_data.shape[1] == 1:
#     # Single category case: create a single-column DataFrame
#     X_encoded = pd.DataFrame(encoded_data.toarray(), columns=['Encoded_' + encoder.get_feature_names_out(['Style'])[0]])
# else:
#     # Multiple categories case: create a DataFrame with multiple columns
#     X_encoded = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Height','Gender', 'Color', 'Style']))


# Concatenate with the original "Height" column
# X = pd.concat([data_imputed[['Height']], X_encoded], axis=1)
X = pd.concat([data_imputed[['Height']], pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Height','Gender', 'Color', 'Style']))], axis=1)
print(f'X: {X}')
print(f'data[Item]: {data["Item"]}')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['Item'], test_size=0.2, random_state=42)


# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)



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


# Example usage
height_input = float(input("Height (in cm): "))
gender_input = input("Gender (M/F): ")
color_input = input("Color: ")
style_input = input("Style (Casual/Formal/Sporty): ")

predicted_item = suggest_fashion_item(height_input,gender_input,color_input,style_input)
print("Suggested Fashion Item:", predicted_item)
