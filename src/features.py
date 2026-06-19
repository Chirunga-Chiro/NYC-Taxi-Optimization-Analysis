import pandas as pd

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-series operational metrics (hour and day name)
    from pickup datetime columns.
    """
    df = df.copy()
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()
    return df

def calculate_tip_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for credit card transactions and builds a tip_percentage metric,
    protecting against division-by-zero errors.
    """
    # Isolate credit card transactions to eliminate cash $0 bias
    cc_df = df[(df['payment_type'] == 1) & (df['fare_amount'] > 0)].copy()
    
    # Calculate target tip percentage
    cc_df['tip_percentage'] = (cc_df['tip_amount'] / cc_df['fare_amount']) * 100
    return cc_df