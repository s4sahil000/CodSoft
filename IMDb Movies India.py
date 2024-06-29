import pandas as pd

# Load the dataset
file_path = 'IMDb Movies India.csv'
movies_df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Display the first few rows of the dataset
movies_df.head()

# Drop rows where the rating is missing
movies_df.dropna(subset=['Rating'], inplace=True)

# Extract the release year from the "Year" column and convert it to numeric
movies_df['Year'] = movies_df['Year'].str.extract(r'(\d{4})')
movies_df['Year'] = pd.to_numeric(movies_df['Year'])

# Convert "Duration" to numeric by removing "min" and converting to integer
movies_df['Duration'] = movies_df['Duration'].str.replace(' min', '')
movies_df['Duration'] = pd.to_numeric(movies_df['Duration'], errors='coerce')

# Fill missing values for "Duration" with the median duration
median_duration = movies_df['Duration'].median()
movies_df['Duration'].fillna(median_duration, inplace=True)

# Fill missing values for other categorical columns with 'Unknown'
movies_df['Director'].fillna('Unknown', inplace=True)
movies_df['Actor 1'].fillna('Unknown', inplace=True)
movies_df['Actor 2'].fillna('Unknown', inplace=True)
movies_df['Actor 3'].fillna('Unknown', inplace=True)


# Handle multiple genres by creating dummy variables
genres_df = movies_df['Genre'].str.get_dummies(sep=', ')
movies_df = pd.concat([movies_df, genres_df], axis=1)

# One-hot encode the director and actors
director_dummies = pd.get_dummies(movies_df['Director'], prefix='Director')
actor1_dummies = pd.get_dummies(movies_df['Actor 1'], prefix='Actor1')
actor2_dummies = pd.get_dummies(movies_df['Actor 2'], prefix='Actor2')
actor3_dummies = pd.get_dummies(movies_df['Actor 3'], prefix='Actor3')

# Concatenate the dummy variables with the main dataframe
movies_df = pd.concat([movies_df, director_dummies, actor1_dummies, actor2_dummies, actor3_dummies], axis=1)

# Drop the original categorical columns
movies_df.drop(columns=['Name', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Select features and target variable
features = movies_df.drop(columns=['Rating'])
target = movies_df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {mse ** 0.5}')

