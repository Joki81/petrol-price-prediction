import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

df = pd.read_csv("/Users/jonathankipping/code/Petrol/price_history.csv")

print(df.head())
print(df.info())
print(df.describe())

df = df.drop(columns=['id', 'node_id', 'source_updated_at'])

X = df.drop(columns=['price_pence'])
y = df['price_pence']

X['recorded_at'] = pd.to_datetime(X['recorded_at'])
X['year'] = X['recorded_at'].dt.year
X['month'] = X['recorded_at'].dt.month
X['day'] = X['recorded_at'].dt.day
X['weekday'] = X['recorded_at'].dt.weekday
X = X.drop(columns=['recorded_at'])

X = pd.get_dummies(X, columns=['fuel_type'], drop_first=True)

num_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'  # fast for large datasets
)

model.fit(X_train, y_train)

# -------------------------
# Predict price for a given date
# -------------------------

def predict_price(model, scaler, date_str, fuel_type='E10'):
    """
    Predict the petrol price (in pence) for a given date and fuel type using a trained model.

    Parameters
    ----------
    model : object
        The trained regression model (e.g., XGBoost or RandomForest) used for prediction.
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler used to scale numeric features during training.
    date_str : str
        The target date for prediction, in 'YYYY-MM-DD' format.
    fuel_type : str, optional (default='E10')
        The fuel type to predict (must match one-hot encoded columns in training data,
        e.g., 'E10', 'B7_STANDARD', etc.).

    Returns
    -------
    float
        The predicted petrol price (in pence) for the specified date and fuel type.

    Notes
    -----
    - The function automatically extracts numeric features from the date:
        year, month, day, and weekday.
    - Fuel type is encoded using the same one-hot columns as the training data.
    - Only numeric features that were scaled during training are transformed with the scaler.
    - The function ensures the feature order matches the training set before prediction.
    """

    date = pd.to_datetime(date_str)

    # Create future dataframe with numeric features
    future_df = pd.DataFrame({
        'year': [date.year],
        'month': [date.month],
        'day': [date.day],
        'weekday': [date.weekday()]
    })

    # Add one-hot encoded fuel columns
    for col in [c for c in X.columns if c.startswith('fuel_type_')]:
        future_df[col] = 1 if col == f'fuel_type_{fuel_type}' else 0

    # Add any missing columns from training set
    for col in X.columns:
        if col not in future_df.columns:
            future_df[col] = 0
    future_df = future_df[X.columns]  # reorder columns

    # Only scale the numeric columns used in training
    numeric_cols = [c for c in num_cols if c in future_df.columns]
    future_df[numeric_cols] = scaler.transform(future_df[numeric_cols])

    # Predict
    predicted_price = model.predict(future_df)[0]
    return predicted_price

# Example usage
date_input = "2026-12-31"
fuel_input = "E10"
predicted = predict_price(model, scaler, date_input, fuel_input)
print(f"Predicted price for {fuel_input} on {date_input}: {predicted:.2f} pence")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # manual RMSE

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

future_date = pd.to_datetime("2026-04-15")
future_df = pd.DataFrame({
    'year': [future_date.year],
    'month': [future_date.month],
    'day': [future_date.day],
    'weekday': [future_date.weekday()],
    'fuel_type_B7_STANDARD': [0],  # One-hot: adjust for your target fuel type
    'fuel_type_E10': [1]           # Example: predicting E10
})

# Ensure all columns match the trained data
for col in X.columns:
    if col not in future_df.columns:
        future_df[col] = 0
future_df = future_df[X.columns]  # Reorder columns

# Scale numeric features
future_df[num_cols] = scaler.transform(future_df[num_cols])

# Predict
future_price = model.predict(future_df)
print(f"Predicted price (pence) for 2026-04-15: {future_price[0]:.2f}")
