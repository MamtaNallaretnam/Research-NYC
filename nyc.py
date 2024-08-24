# %%
import pandas as pd
import pyarrow.parquet as pq

# %%
parquet_files = [   "/home/shives/mamta_research/data/yellow_tripdata_2019-01.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2019-02.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2019-03.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2019-04.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2019-12.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2020-01.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2020-02.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2020-03.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2020-04.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2020-05.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2020-10.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2020-11.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2021-06.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2021-07.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2021-08.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2021-09.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2021-10.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2021-11.parquet",
                    "/home/shives/mamta_research/data/yellow_tripdata_2021-12.parquet"]

dataframes = []

for file in parquet_files:
    df = pq.read_table(file).to_pandas()
    dataframes.append(df)


# %%
final_df = pd.concat(dataframes)

# %%
final_df

# %%
# Convert 'tpep_pickup_datetime' and 'tpep_dropoff_datetime' to datetime type
final_df['tpep_pickup_datetime'] = pd.to_datetime(final_df['tpep_pickup_datetime'])
final_df['tpep_dropoff_datetime'] = pd.to_datetime(final_df['tpep_dropoff_datetime'])



# %%
# Calculate time difference in seconds between pickup and dropoff
final_df['time_diff_sec'] = (final_df['tpep_dropoff_datetime'] - final_df['tpep_pickup_datetime']).dt.total_seconds()

# %%

# Extract features from datetime column
final_df['pickup_date'] = final_df['tpep_pickup_datetime'].dt.date

# %%
final_df['dropoff_date'] = final_df['tpep_dropoff_datetime'].dt.date

# %%
final_df.drop('tpep_pickup_datetime', axis=1, inplace=True)


# %%
final_df.drop('tpep_dropoff_datetime', axis=1, inplace=True)

# %%
final_df

# %%
# Convert 'pickup_date' column to datetime type
final_df['pickup_date'] = pd.to_datetime(final_df['pickup_date'])

# %%
# date_year_pickup = final_df['pickup_date'].dt.year
# date_month_pickup = final_df['pickup_date'].dt.month
# date_day_pickup = final_df['pickup_date'].dt.day

# %%
# final_df.drop('pickup_date', axis=1, inplace=True)

# %%
# Convert 'dropoff_date' column to datetime type
final_df['dropoff_date'] = pd.to_datetime(final_df['dropoff_date'])

# %%

# date_year_dropoff = final_df['dropoff_date'].dt.year
# date_month_dropoff = final_df['dropoff_date'].dt.month
# date_day_dropoff = final_df['dropoff_date'].dt.day


# %%
# final_df.drop('dropoff_date', axis=1, inplace=True)

# %%
# Separate numerical and categorical columns
numerical_cols = final_df.select_dtypes(include=['number']).columns
categorical_cols = final_df.select_dtypes(include=['object']).columns

# Display separated numerical and categorical data
numerical_data_test = final_df[numerical_cols]
categorical_data_test = final_df[categorical_cols]

# %%
import numpy as np

# %%
final_df.replace ('N', np.nan, inplace = True)
col_missing_values = final_df.columns[final_df.isnull().any()].tolist()
print(col_missing_values)

# %%
final_df.replace ('None', np.nan, inplace = True)
col_missing_values = final_df.columns[final_df.isnull().any()].tolist()
print(col_missing_values)

# %%
# Find columns with NaN values
columns_with_nan = final_df.columns[final_df.isnull().any()].tolist()
print("Columns with NaN values: ", columns_with_nan)

# %%
numerical_cols_with_nan = []
categorical_cols_with_nan = []

for column in columns_with_nan:
    if final_df[column].dtype == 'object':
        categorical_cols_with_nan.append(column)
    else:
        numerical_cols_with_nan.append(column)

print("Numerical columns with NaN values: ", numerical_cols_with_nan)
print("Categorical columns with NaN values: ", categorical_cols_with_nan)

# %%
#imputing for numerical data set
from sklearn.impute import SimpleImputer

# Impute missing values in numerical columns with mean
numerical_imputer = SimpleImputer(strategy='mean')
final_df[numerical_cols_with_nan] = numerical_imputer.fit_transform(final_df[numerical_cols_with_nan])

# %%
#imputing for categorical data set
# Impute missing values in categorical columns with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
final_df[categorical_cols_with_nan] = categorical_imputer.fit_transform(final_df[categorical_cols_with_nan])

# %%
# Find columns with NaN values
columns_with_nan = final_df.columns[final_df.isnull().any()].tolist()
print("Columns with NaN values: ", columns_with_nan)

# %%
final_df.drop(['store_and_fwd_flag'], axis=1, inplace=True)

# %%
# Filter data for X_test_1 based on date range
X_test_1 = final_df[(final_df['pickup_date'] >= pd.to_datetime('2019-01-01')) & 
                    (final_df['pickup_date'] < pd.to_datetime('2019-04-30'))]

# Extract y_test_1
y_test_1 = X_test_1['time_diff_sec']

# Drop unnecessary columns
X_test_1 = X_test_1.drop(columns=['pickup_date', 'time_diff_sec'])

# %%
X_test_1

# %%
X_test_2 = final_df[(final_df['pickup_date'] >= pd.to_datetime('2019-12-01')) & 
                    (final_df['pickup_date'] < pd.to_datetime('2020-05-31'))]
y_test_2 = X_test_2['time_diff_sec']
X_test_2 = X_test_2.drop(columns=['pickup_date', 'time_diff_sec'])

# %%
X_test_3 = final_df[(final_df['pickup_date'] >= pd.to_datetime('2020-10-01')) & 
                    (final_df['pickup_date'] < pd.to_datetime('2020-11-30'))]
y_test_3 = X_test_3['time_diff_sec']
X_test_3 = X_test_3.drop(columns=['pickup_date', 'time_diff_sec'])

# %%
X_test_4 = final_df[(final_df['pickup_date'] >= pd.to_datetime('2021-06-01')) & 
                    (final_df['pickup_date'] < pd.to_datetime('2021-12-31'))]
y_test_4 = X_test_4['time_diff_sec']
X_test_4 = X_test_4.drop(columns=['pickup_date', 'time_diff_sec'])

# %%
final_df

# %%
# Find columns with NaN values
columns_with_nan = final_df.columns[final_df.isnull().any()].tolist()
print("Columns with NaN values: ", columns_with_nan)


# %%
final_df.dtypes

# %%

from sklearn.model_selection import train_test_split
y = final_df['time_diff_sec']
X = final_df.drop(columns = ['time_diff_sec'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
pip install xgboost

# %%
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
# Initialize XGBRegressor
estimator = XGBRegressor()

# Initialize RFE with XGBRegressor
rfe = RFE(estimator, n_features_to_select=1)

# Fit RFE model
rfe.fit(X_train, y_train)

# %%
rank = {}

for i in range(0,len(rfe.ranking_)):
  rank[rfe.ranking_[i]] = X_train.columns[i]

# %%
size = len(rank) + 1

for i in range(1,len(rank)+1):
  print(f'{i}: {rank[i]}')

# %%
importance = {}

for i in range(0,len(rfe.ranking_)):
  importance[rfe.ranking_[i]] = X_train.columns[i]

# %%
size = len(importance) + 1

for i in range(1,len(importance)+1):
  print(f'{i}: {importance[i]}')

# %%
accuracy = []
features = []
models = {}

for i in range(1,size):
  features.append(importance[i])

  model = XGBRegressor()
  model.fit(X_train[features],y_train)

  models[i] = model

# %%
from sklearn.metrics import mean_absolute_error

# %%
min_mae_n = None  # Initialize to None
mae_values = []

for i in range(1, size):
    x_valid = X_test[features[0:i]]
    y_valid = y_test
    y_pred = models[i].predict(x_valid)

    # Calculate Mean Abosulte Error (MAE)
    mae = mean_absolute_error(y_valid, y_pred)
    mae_values.append(mae)

    # Update min_mae_n if current MAE is lower
    if min_mae_n is None or mae < mae_values[min_mae_n-1]:
        min_mae_n = i

    print(f'{i}: {mae}')

# %%
import matplotlib.pyplot as plt


# %%
plt.plot(range(1, size), mae_values, marker='o', linestyle='-')
plt.xlabel('number of features')
plt.ylabel('MAE')
plt.title('MAE vs Features')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Print features corresponding to minimum MAE
if min_mae_n is not None:
    print("Features corresponding to minimum MAE:")
    print(features[0:min_mae_n])
else:
    print("No minimum MAE found.")

# %%
XGBRegressor() # looking at hyperparameters

# %%
from sklearn.model_selection import RandomizedSearchCV

# Define your parameter search space
params = {
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3, 4, 5],
    'max_depth': [5, 6, 7, 8, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

# Create your XGBoost regressor model with MAE objective
model = XGBRegressor(objective='reg:squarederror')  # Change to 'reg:linear' for MAE

# Define the RandomizedSearchCV object with MAE scoring metric
random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=100, cv=5, scoring='neg_mean_absolute_error', random_state=42)

# Fit the model with hyperparameter tuning
random_search.fit(X_train[features[0:min_mae_n]], y_train)

# Access the best hyperparameters
best_params = random_search.best_params_

# Print the best parameters
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = random_search.best_estimator_

# Use the best_model for prediction
y_pred = best_model.predict(X_test[features[0:min_mae_n]])

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

# Print MAE
print("Mean Absolute Error:", mae)

# %%
params = {
    'subsample': 0.9,
    'min_child_weight': 5,
    'max_depth': 5,
    'learning_rate': 0.1,
    'colsample_bytree': 0.6
}

xgb_regressor = XGBRegressor(**params)

# Fit the regressor to the training data
xgb_regressor.fit(X_train[features[0:min_mae_n]], y_train)

# Predict on the testing data
y_pred = xgb_regressor.predict(X_test[features[0:min_mae_n]])

# Calculate Mean Absolute Error (MAE) to evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print("Mean absoulte Error:", mae)

# %%
pip install shap

# %%
import shap

# %%
import xgboost as xgb

# Wrap the XGBoost model inside a callable function
def fit_xgb_model(X_train, y_train):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Define a function that internally predicts using the fitted model
    def model_predict(X):
        return model.predict(X)

    return model_predict

# Fit the XGBoost model
model_predict = fit_xgb_model(X_train, y_train)

# Calculate SHAP values
explainer = shap.Explainer(model_predict, X_train)
shap_values = explainer.shap_values(X_test)

# %%
# Visualize SHAP values
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

# %% [markdown]
# INCREMENTAL LEARNING 

# %%
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
#incremental learning for X_test_1
# split data into training and testing sets
# then split training set in half
X_train, X_test, y_train, y_test = train_test_split(X_test_1, y_test_1, test_size=0.1, random_state=0)
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(
    X_train, y_train, test_size=0.5, random_state=0
)

xg_train_1 = xgb.DMatrix(X_train_1, label=y_train_1)
xg_train_2 = xgb.DMatrix(X_train_2, label=y_train_2)
xg_test = xgb.DMatrix(X_test, label=y_test)

params = {'objective': 'reg:squarederror', 'verbose': False}
model_1 = xgb.train(params, xg_train_1, 30)
model_1.save_model('model_1.model')

# ================= train two versions of the model =====================#
model_2_v1 = xgb.train(params, xg_train_2, 30)
model_2_v2 = xgb.train(params, xg_train_2, 30, xgb_model='model_1.model')

print(mean_squared_error(model_1.predict(xg_test), y_test))     # benchmark
print(mean_squared_error(model_2_v1.predict(xg_test), y_test))  # "before"
print(mean_squared_error(model_2_v2.predict(xg_test), y_test))  # "after"

# %%
#incremental learning for X_test_2

X_train, X_test, y_train, y_test = train_test_split(X_test_2, y_test_2, test_size=0.1, random_state=0)
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(
    X_train, y_train, test_size=0.5, random_state=0
)

xg_train_1 = xgb.DMatrix(X_train_1, label=y_train_1)
xg_train_2 = xgb.DMatrix(X_train_2, label=y_train_2)
xg_test = xgb.DMatrix(X_test, label=y_test)

params = {'objective': 'reg:squarederror', 'verbose': False}
model_1 = xgb.train(params, xg_train_1, 30)
model_1.save_model('model_1.model')


model_2_v1 = xgb.train(params, xg_train_2, 30)
model_2_v2 = xgb.train(params, xg_train_2, 30, xgb_model='model_1.model')

print(mean_squared_error(model_1.predict(xg_test), y_test))     # benchmark
print(mean_squared_error(model_2_v1.predict(xg_test), y_test))  # "before"
print(mean_squared_error(model_2_v2.predict(xg_test), y_test))  # "after"

# %%
#incremental learning for X_test_3

X_train, X_test, y_train, y_test = train_test_split(X_test_3, y_test_3, test_size=0.1, random_state=0)
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(
    X_train, y_train, test_size=0.5, random_state=0
)

xg_train_1 = xgb.DMatrix(X_train_1, label=y_train_1)
xg_train_2 = xgb.DMatrix(X_train_2, label=y_train_2)
xg_test = xgb.DMatrix(X_test, label=y_test)

params = {'objective': 'reg:squarederror', 'verbose': False}
model_1 = xgb.train(params, xg_train_1, 30)
model_1.save_model('model_1.model')


model_2_v1 = xgb.train(params, xg_train_2, 30)
model_2_v2 = xgb.train(params, xg_train_2, 30, xgb_model='model_1.model')

print(mean_squared_error(model_1.predict(xg_test), y_test))     # benchmark
print(mean_squared_error(model_2_v1.predict(xg_test), y_test))  # "before"
print(mean_squared_error(model_2_v2.predict(xg_test), y_test))  # "after"

# %%
#incremental learning for X_test_4

X_train, X_test, y_train, y_test = train_test_split(X_test_4, y_test_4, test_size=0.1, random_state=0)
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(
    X_train, y_train, test_size=0.5, random_state=0
)

xg_train_1 = xgb.DMatrix(X_train_1, label=y_train_1)
xg_train_2 = xgb.DMatrix(X_train_2, label=y_train_2)
xg_test = xgb.DMatrix(X_test, label=y_test)

params = {'objective': 'reg:squarederror', 'verbose': False}
model_1 = xgb.train(params, xg_train_1, 30)
model_1.save_model('model_1.model')


model_2_v1 = xgb.train(params, xg_train_2, 30)
model_2_v2 = xgb.train(params, xg_train_2, 30, xgb_model='model_1.model')

print(mean_squared_error(model_1.predict(xg_test), y_test))     # benchmark
print(mean_squared_error(model_2_v1.predict(xg_test), y_test))  # "before"
print(mean_squared_error(model_2_v2.predict(xg_test), y_test))  # "after"


