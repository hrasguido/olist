import pandas as pd

def clean_order_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'])

    return df

def clean_order_payments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df['payment_installments']!=0]

    notdefined_0 = df[(df['payment_value']==0) & (df['payment_type']=='not_defined')]
    df.drop(notdefined_0.index, inplace=True)

    df['payment_type'] = df['payment_type'].str.replace('_', ' ').str.title()

    return df

def clean_order_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=['review_comment_title', 'review_comment_message'])
    
    return df

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df['order_status'] == 'delivered']

    df = df.dropna()   

    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
    df['order_delivered_carrier_date'] = pd.to_datetime(df['order_delivered_carrier_date'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])

    return df

def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.dropna(inplace=True)
    df.drop(df[df['product_weight_g']==0].index, inplace=True)

    return df