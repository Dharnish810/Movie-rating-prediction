import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv(r"C:\Users\Dharnish\Downloads\IMDb Movies India.csv\IMDb Movies India.csv", encoding='ISO-8859-1')

# Display initial info
print(df.info())
print(df.describe())

# Check for missing values and duplicates
print("Missing values:", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

# Data Cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Convert data types
df['Year'] = df['Year'].str.replace(r'\(|\)', '', regex=True).astype(int)
df['Duration'] = pd.to_numeric(df['Duration'].str.replace(' min', ''))
df['Votes'] = pd.to_numeric(df['Votes'].str.replace(',', ''))

# Feature Engineering: Average ratings for genre, director, and actors
df['Genre_Average_Rating'] = df.groupby('Genre')['Rating'].transform('mean')
df['Director_Average_Rating'] = df.groupby('Director')['Rating'].transform('mean')
df['Actor1_Average_Rating'] = df.groupby('Actor 1')['Rating'].transform('mean')
df['Actor2_Average_Rating'] = df.groupby('Actor 2')['Rating'].transform('mean')
df['Actor3_Average_Rating'] = df.groupby('Actor 3')['Rating'].transform('mean')

# Visualize data
plt.figure(figsize=(10, 5))
sns.histplot(df['Rating'], bins=30, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Split dataset into features and target variable
X = df[['Year', 'Votes', 'Duration', 'Genre_Average_Rating', 'Director_Average_Rating', 
         'Actor1_Average_Rating', 'Actor2_Average_Rating', 'Actor3_Average_Rating']]
y = df['Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train models
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Model Evaluation
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\n{model_name} Performance:')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R^2 Score: {r2:.2f}')

evaluate_model(y_test, y_pred_lr, 'Linear Regression')
evaluate_model(y_test, y_pred_rf, 'Random Forest')

# Scatter plots for predictions
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.title('Linear Regression: Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Line of equality
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.title('Random Forest: Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Line of equality
plt.show()

# Trial Data Prediction
trial_data = pd.DataFrame({
    'Year': [2018],
    'Votes': [100],
    'Duration': [130],
    'Genre_Average_Rating': [6.0],
    'Director_Average_Rating': [6.5],
    'Actor1_Average_Rating': [5.5],
    'Actor2_Average_Rating': [5.8],
    'Actor3_Average_Rating': [5.3]
})

predicted_rating_rf = rf_model.predict(trial_data)
print('Predicted Rating for trial data (Random Forest):', predicted_rating_rf[0])
