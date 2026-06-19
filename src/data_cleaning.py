import pandas as pd

def clean_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw NYC Yellow Taxi Parquet data by removing impossible rows,
    extreme noise, and legal baseline anomalies.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")
        
    # Apply your validated filters
    cleaned_df = df[
        (df['trip_distance'] > 0) & (df['trip_distance'] < 50) &
        (df['fare_amount'] >= 2.50) & (df['fare_amount'] <= 200) &
        (df['passenger_count'] > 0)
    ].copy()
    
    return cleaned_df