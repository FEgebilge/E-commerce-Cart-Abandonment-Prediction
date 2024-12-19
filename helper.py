import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from memory_profiler import memory_usage

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

def visualize_preprocessed_data(preprocessed_file):
    # Load preprocessed dataset
    df = pd.read_csv(preprocessed_file)
    
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle('Visualizations of Preprocessed Dataset', fontsize=16, y=0.92)

    # 1. Distribution of the Target Variable
    sns.countplot(x='abandoned', data=df, ax=axes[0, 0], palette='Set2', hue='abandoned')
    axes[0, 0].set_title('Target Variable Distribution')
    axes[0, 0].set_xlabel('Abandoned (0: Not Abandoned, 1: Abandoned)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend([], [], frameon=False)  # Remove the redundant legend

    # 2. Distribution of Key Numerical Features (price)
    sns.histplot(data=df, x='price', hue='abandoned', kde=True, palette='coolwarm', bins=30, ax=axes[0, 1])
    axes[0, 1].set_title('Price Distribution by Abandonment')
    axes[0, 1].set_xlabel('Price')
    axes[0, 1].set_ylabel('Count')

    # 3. Correlation Heatmap
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1, 0], cbar_kws={'shrink': 0.8})
    axes[1, 0].set_title('Correlation Heatmap')

    # 4. Boxplot of grand_total
    sns.boxplot(data=df, x='abandoned', y='grand_total', palette='coolwarm', ax=axes[1, 1], hue='abandoned', dodge=False)
    axes[1, 1].set_title('Grand Total Distribution by Abandonment')
    axes[1, 1].set_xlabel('Abandoned (0: Not Abandoned, 1: Abandoned)')
    axes[1, 1].set_ylabel('Grand Total')
    axes[1, 1].legend([], [], frameon=False)  # Remove redundant legend

    # 5. Distribution of total_purchases
    sns.histplot(data=df, x='total_purchases', hue='abandoned', kde=True, palette='coolwarm', bins=30, ax=axes[2, 0])
    axes[2, 0].set_title('Total Purchases Distribution by Abandonment')
    axes[2, 0].set_xlabel('Total Purchases')
    axes[2, 0].set_ylabel('Count')

    # 6. Boxplot of total_orders
    sns.boxplot(data=df, x='abandoned', y='total_orders', palette='coolwarm', ax=axes[2, 1], hue='abandoned', dodge=False)
    axes[2, 1].set_title('Total Orders Distribution by Abandonment')
    axes[2, 1].set_xlabel('Abandoned (0: Not Abandoned, 1: Abandoned)')
    axes[2, 1].set_ylabel('Total Orders')
    axes[2, 1].legend([], [], frameon=False)  # Remove redundant legend

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Track training time and memory usage
    def train_model():
        model.fit(X_train, y_train)

    training_time = time.time()
    mem_usage = memory_usage(train_model, interval=0.1)
    training_time = time.time() - training_time

    # Evaluate model
    start_eval = time.time()
    y_pred = model.predict(X_test)
    evaluation_time = time.time() - start_eval

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Performance of {model_name}:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  Training Time: {training_time:.2f} seconds")
    print(f"  Evaluation Time: {evaluation_time:.2f} seconds")
    print(f"  Memory Used: {max(mem_usage) - min(mem_usage):.2f} MB")
    print(f"  Confusion Matrix:\n{cm}\n")

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Training Time (s)': training_time,
        'Evaluation Time (s)': evaluation_time,
        'Memory Used (MB)': max(mem_usage) - min(mem_usage),
        'Confusion Matrix': cm
    }