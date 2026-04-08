import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("/Users/jonathankipping/code/Petrol/price_history.csv")

# Inspect
print(df.head())
print(df.info())
print(df.describe())

# Correlation heatmap (numeric only)
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True)
plt.show()

# Handle missing values (choose ONE)
df = df.dropna()

# Define Features
X = df.drop("price_pence", axis=1)  # change "price" to your column name
y = df["price_pence"]

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale Features:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Performance:
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)
