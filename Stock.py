import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('housing_data.csv')  # Replace 'housing_data.csv' with your dataset
    return data

data = load_data()

# Sidebar for user input
st.sidebar.header('User Input')
feature = st.sidebar.selectbox('Select a feature:', data.columns)

# Descriptive statistics
st.subheader('Descriptive Statistics')
st.write(data.describe())

# Data visualization
st.subheader('Data Visualization')

# Histogram of the selected feature
plt.figure(figsize=(8, 6))
sns.histplot(data[feature], bins=20, kde=True)
plt.xlabel(feature)
plt.ylabel('Frequency')
st.pyplot()

# Scatter plot of housing prices vs the selected feature
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data[feature], y=data['price'])
plt.xlabel(feature)
plt.ylabel('Price')
st.pyplot()

# Regression analysis
st.subheader('Regression Analysis')

# Splitting the data into training and testing sets
X = data[[feature]]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
st.write('Coefficient:', model.coef_[0])
st.write('Intercept:', model.intercept_)
st.write('R-squared:', model.score(X_test, y_test))

# Streamlit web application
st.title('Housing Price Prediction App')

# Input feature value from user
input_value = st.number_input(f'Enter the {feature}:')

# Predict housing price
predicted_price = model.predict([[input_value]])

st.write(f'Predicted price: ${predicted_price[0]:,.2f}')

