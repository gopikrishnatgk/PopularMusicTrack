#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import pandas as pd

# Loading/reading data
music_data = pd.read_csv("music.csv")

# Printing initial data from dataset
music_data.head()

# Dropping unwanted columns and moving forward
music_data.drop(columns = ['Unnamed: 0'], inplace = True)

music_data.head()

# Dataset information
music_data.info()

# Exploiratory Data Analysis (EDA)

# Importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# List of column names
list(music_data)

# Selecting all columns(features) except target variable(Popularity)
variables = music_data.loc[:, music_data.columns != 'Popularity']
list(variables)

# Selecting features
features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness']

for feature in features:
    plt.figure(figsize = (7,5))
    sns.scatterplot(data = music_data, x = feature, y = 'Popularity')
    plt.title('Popularity vs {feature}')
    plt.show()

# Finding correlation between features using correlation matrix
num_cols = music_data.select_dtypes(include = ['float64', 'int64']).columns
num_data = music_data[num_cols]

corr_matrix = num_data.corr()

plt.figure(figsize = (10,8))
sns.heatmap(corr_matrix, annot=True, cmap = 'coolwarm', fmt = '.2f', linewidths = 0.5)
plt.title('Correlation matrix')
plt.show()

# With the above map, we can say music popularity is highly correlated with loudness and danceability

# Features distribution graphs
for feature in features:
    plt.figure(figsize = (5, 3))
    sns.histplot(music_data[feature], kde = True)
    plt.title(f'{feature} distribution')
    plt.show()

# Distribution of 'Energy' feature is most likely a bell shaped

# Train model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Selecting features and terget
X = music_data[features]
y = music_data['Popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature normalization
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Parameter grid for hyperparameter tuning
para_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Hyperparameter tuning using GridSearchCV for RandomForestRegressor
gs_rf = GridSearchCV(RandomForestRegressor(random_state = 42), para_grid, refit = True, verbose = 2, cv = 5)
gs_rf.fit(X_train_sc, y_train)
best_para_rf = gs_rf.best_params_
best_rf_model = gs_rf.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_sc)

# Predictions
y_pred = best_rf_model.predict(X_test_sc)

plt.figure(figsize = (5,3))
plt.scatter(y_test, y_pred, alpha = 0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linewidth = 3)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Prediction with Random Forest Algorithm')
plt.show()

# We can observe that most of the data points are present nearer to regression line
# We can say our is perfoming good

# Model performance
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(y_test, y_pred, squared=False))
print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(y_test, y_pred))
print('Explained Variance Score:', metrics.explained_variance_score(y_test, y_pred))
print('Max Error:', metrics.max_error(y_test, y_pred))
print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_test, y_pred))
print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_pred))
print('R^2:', metrics.r2_score(y_test, y_pred))
print('Mean Poisson Deviance:', metrics.mean_poisson_deviance(y_test, y_pred))
print('Mean Gamma Deviance:', metrics.mean_gamma_deviance(y_test, y_pred))
