import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_dataset(input_file: str, output_file: str):
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Step 1: Remove Extra Columns (Unnamed columns)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Step 2: Filter rows to include only 'canceled' and 'complete'
    df = df[df['status'].isin(['canceled', 'complete'])]

    # Step 3: Map 'status' to 'abandoned/not abandoned' (1: abandoned, 0: not abandoned)
    df['abandoned'] = df['status'].apply(lambda x: 1 if x == 'canceled' else 0)

    # Step 4: One-hot encode 'category_name_1' (cart contents)
    category_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    category_encoded = category_encoder.fit_transform(df[['category_name_1']])
    category_encoded_df = pd.DataFrame(category_encoded, columns=category_encoder.get_feature_names_out(['category_name_1']))
    df = pd.concat([df, category_encoded_df], axis=1)

    # Step 5: One-hot encode 'payment_method'
    payment_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    payment_encoded = payment_encoder.fit_transform(df[['payment_method']])
    payment_encoded_df = pd.DataFrame(payment_encoded, columns=payment_encoder.get_feature_names_out(['payment_method']))
    df = pd.concat([df, payment_encoded_df], axis=1)

    # Step 6: Calculate purchase history for each customer
    purchase_history = df.groupby('Customer ID').agg(
        total_purchases=('grand_total', 'sum'),
        total_orders=('Customer ID', 'count')
    ).reset_index()
    df = df.merge(purchase_history, on='Customer ID', how='left')

    # Step 7: Drop unnecessary columns
    columns_to_drop = [
        'status', 'created_at', 'sku', 'increment_id', 'category_name_1',
        'sales_commission_code', 'Working Date', 'BI Status', ' MV ',
        'Year', 'Month', 'Customer Since', 'M-Y', 'FY', 'Customer ID', 'payment_method'
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    # Step 8: Handle missing values
    df.fillna(0, inplace=True)  # Replace NaN with 0 (can be customized as needed)

    # Step 9: Feature Scaling
    scaler = MinMaxScaler()
    numerical_features = ['price', 'qty_ordered', 'grand_total', 'discount_amount', 'total_purchases', 'total_orders']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Step 10: Investigate and Handle Outliers (e.g., Clipping to 95th percentile)
    for col in numerical_features:
        upper_limit = df[col].quantile(0.95)
        df[col] = df[col].clip(upper=upper_limit)

    # Step 11: Balance Target Variable using Undersampling
    abandoned_df = df[df['abandoned'] == 1]
    not_abandoned_df = df[df['abandoned'] == 0].sample(n=len(abandoned_df), random_state=42)
    df_balanced = pd.concat([abandoned_df, not_abandoned_df])

    # Step 12: Save the preprocessed dataset
    df_balanced.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to {output_file}")
