import pandas as pd
from typing import List


DEVICE_TYPES = ['Desktop', 'Tablet', 'Mobile', 'Laptop']
PAYMENT_METHODS = ['Mobile Payment', 'Credit Card', 'Gift Card', 'Debit Card', 'Other']



def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Transaction_Time is '04:04:15' in your data
    df['Hour'] = pd.to_datetime(df['Transaction_Time'], format='%H:%M:%S').dt.hour
    df['Is_Night_Transaction'] = ((df['Hour'] >= 0) & (df['Hour'] <= 5)).astype(int)
    return df


def encode_loyalty_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    loyalty_map = {
        'Bronze': 1, 'Silver': 2, 'Gold': 3, 
        'Platinum': 4, 'VIP': 5, 'Unknown': 0
    }
    # Fill NaN with 'Unknown' before mapping
    df['Customer_Loyalty_Tier'] = df['Customer_Loyalty_Tier'].fillna('Unknown')
    df['Loyalty_Score'] = df['Customer_Loyalty_Tier'].map(loyalty_map).fillna(0)
    return df


def one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for device in DEVICE_TYPES:
        df[f'Device_Type_{device}'] = (df['Device_Type'] == device).astype(int)
    for payment in PAYMENT_METHODS:
        df[f'Payment_Method_{payment}'] = (df['Payment_Method'] == payment).astype(int)
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Customer_Age'] = df['Customer_Age'].fillna(median_age := df['Customer_Age'].median())
    df['Purchase_to_Age'] = df['Purchase_Amount'] / (df['Customer_Age'] + 1)
    return df


def drop_non_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    # We drop these because XGBoost only understands numbers
    cols_to_drop = [
        'Transaction_ID', 'Customer_ID', 'Transaction_Date', 'Transaction_Time',
        'Location', 'Store_ID', 'IP_Address', 'Product_SKU', 'Product_Category',
        'Customer_Loyalty_Tier', 'Device_Type', 'Payment_Method', 'Fraud_Flag'
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = extract_time_features(df)
    df = encode_loyalty_tier(df)
    df = one_hot_encode_categoricals(df)
    df = create_derived_features(df)
    # Footfall_Count is already numeric, so it stays!
    df = drop_non_model_columns(df)
    return df


if __name__ == "__main__":
    print("Testing pipeline...")
    # Just a quick print to make sure the math works
    test_df = pd.DataFrame([{
        'Transaction_Time': '04:00:00', 'Customer_Age': 25, 
        'Purchase_Amount': 100, 'Customer_Loyalty_Tier': 'Silver',
        'Device_Type': 'Mobile', 'Payment_Method': 'Credit Card',
        'Footfall_Count': 300
    }])
    print(preprocess_features(test_df).columns)