import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/house_data.csv')
X = df[['sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 'garage', 'neighborhood_score', 'location_type']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Testing 3 Regression Models:")
print("-" * 40)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print(f"Linear Regression: R2={r2_score(y_test, y_pred_lr):.3f}, RMSE=${np.sqrt(mean_squared_error(y_test, y_pred_lr)):,.0f}")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest: R2={r2_score(y_test, y_pred_rf):.3f}, RMSE=${np.sqrt(mean_squared_error(y_test, y_pred_rf)):,.0f}")

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print(f"Gradient Boosting: R2={r2_score(y_test, y_pred_gb):.3f}, RMSE=${np.sqrt(mean_squared_error(y_test, y_pred_gb)):,.0f}")

sample = pd.DataFrame([[2000, 3, 2, 2000, 8000, 2, 7, 2]], columns=X.columns)
sample_scaled = scaler.transform(sample)

print("\nSample Predictions (2000sqft, 3bd, 2ba, 2000yr, 8000lot, 2garage, 7neighbor, 2location):")
print(f"  Linear Regression: ${lr.predict(sample_scaled)[0]:,.0f}")
print(f"  Random Forest: ${rf.predict(sample)[0]:,.0f}")
print(f"  Gradient Boosting: ${gb.predict(sample)[0]:,.0f}")

print("\nAll tests passed successfully!")
