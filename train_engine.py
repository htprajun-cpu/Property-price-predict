import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def fetch_crime():
    url = "https://api.worldbank.org/v2/country/all/indicator/VC.IHR.PSRC.P5?format=json&per_page=1000"
    try:
        data = requests.get(url).json()[1]
        return {item['country']['value']: item['value'] for item in data if item['value']}
    except: return {}

def train():
    df = pd.read_csv('global_master_geocoded.csv').dropna(subset=['lat', 'lon'])
    crime_map = fetch_crime()
    
    # Injecting World Bank Crime Data (Example: India)
    df['crime_rate'] = crime_map.get('India', 5.0) 
    
    # Accuracy Features
    X = df[['area_sqft', 'bhk', 'lat', 'lon', 'crime_rate']]
    y = np.log1p(df['price_lakhs']) 

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', HistGradientBoostingRegressor(max_iter=1000))
    ])
    
    model.fit(X, y)
    joblib.dump(model, 'global_precision_agent.joblib')
    print("✅ Model trained and saved!")

if __name__ == "__main__":
    train()