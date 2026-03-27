import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_loading():
    df = pd.read_csv('data/house_data.csv')
    assert len(df) > 0, "Dataset should not be empty"
    assert 'price' in df.columns, "Dataset should have price column"
    print("✓ Data loading test passed")

def test_feature_columns():
    df = pd.read_csv('data/house_data.csv')
    expected_features = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 'garage', 'neighborhood_score', 'location_type']
    for feature in expected_features:
        assert feature in df.columns, f"Missing feature: {feature}"
    print("✓ Feature columns test passed")

def test_model_training():
    df = pd.read_csv('data/house_data.csv')
    X = df[['sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 'garage', 'neighborhood_score', 'location_type']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.5, f"R² score too low: {r2}"
    print(f"✓ Model training test passed (R² = {r2:.3f})")

def test_prediction():
    df = pd.read_csv('data/house_data.csv')
    X = df[['sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 'garage', 'neighborhood_score', 'location_type']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    sample = [[2000, 3, 2, 2000, 8000, 2, 7, 2]]
    prediction = model.predict(sample)[0]
    
    assert prediction > 0, "Prediction should be positive"
    assert 50000 < prediction < 2000000, f"Prediction seems unrealistic: {prediction}"
    print(f"✓ Prediction test passed (${prediction:,.0f})")

def test_all_models():
    df = pd.read_csv('data/house_data.csv')
    X = df[['sqft', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 'garage', 'neighborhood_score', 'location_type']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': (LinearRegression(), X_train_scaled, X_test_scaled, True),
        'Random Forest': (RandomForestRegressor(n_estimators=10, random_state=42), X_train, X_test, False),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=10, random_state=42), X_train, X_test, False)
    }
    
    for name, (model, X_tr, X_te, scaled) in models.items():
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"  {name}: R²={r2:.3f}, RMSE=${rmse:,.0f}, MAE=${mae:,.0f}")
        assert r2 > 0, f"{name} should have positive R² score"
    
    print("✓ All models test passed")

def run_all_tests():
    print("\n" + "="*50)
    print("Running House Price Predictor Tests")
    print("="*50 + "\n")
    
    try:
        test_data_loading()
        test_feature_columns()
        test_model_training()
        test_prediction()
        test_all_models()
        
        print("\n" + "="*50)
        print("✅ All tests passed successfully!")
        print("="*50 + "\n")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
