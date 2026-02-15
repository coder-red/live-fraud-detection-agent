import pandas as pd
import numpy as np

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371
    d_lat, d_lon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(d_lat / 2)**2 + 
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2)**2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def preprocess_features(df: pd.DataFrame, for_agent: bool = False) -> pd.DataFrame:
    """Feature engineering for fraud detection."""
    df = df.copy()
    
    # Temporal
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 3)).astype(int)
    
    # Demographics & Geospatial
    df['age'] = (df['trans_date_trans_time'] - pd.to_datetime(df['dob'])).dt.days // 365 
    df['dist_to_merchant'] = calculate_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

    if for_agent:
        return df[['category', 'amt', 'job', 'age', 'dist_to_merchant', 'hour', 'gender', 'city', 'state']]
    
    # XGBoost Native Categorical Encoding
    cat_cols = ['category', 'gender', 'state']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    # Selection of features for modeling
    feature_cols = [
        'amt', 'city_pop', 'dist_to_merchant', 'age',
        'hour', 'day_of_week', 'is_weekend', #'is_night',
        'category', 'gender', 'state' # XGBoost will use these as categories
    ]
    
    return df[feature_cols]