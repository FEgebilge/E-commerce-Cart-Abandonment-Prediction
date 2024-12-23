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
    
    # Step 1: Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Step 2: Keep only relevant statuses
    df = df[df['status'].isin(['canceled', 'complete'])]

    # Step 3: Create the target variable ('abandoned')
    df['abandoned'] = df['status'].apply(lambda x: 1 if x == 'canceled' else 0)

    # Step 4: Aggregate features at the cart level (increment_id)
    cart_features = df.groupby('increment_id').agg(
        total_price=('grand_total', 'sum'),
        total_items=('qty_ordered', 'sum'),
        total_discount=('discount_amount', 'sum'),
        num_unique_categories=('category_name_1', 'nunique'),
        abandoned=('abandoned', 'max'),  # Abandoned cart if any item is canceled
        payment_method=('payment_method', 'first'),  # Assuming one payment method per cart
        customer_id=('Customer ID', 'first')  # Customer ID associated with the cart
    ).reset_index()

    # Step 5: Remove rows with non-numeric values in numeric columns
    numeric_cols = ['total_price', 'total_items', 'total_discount', 'num_unique_categories']
    for col in numeric_cols:
        cart_features[col] = pd.to_numeric(cart_features[col], errors='coerce')
    cart_features = cart_features.dropna(subset=numeric_cols)

    # Step 6: Create binary flags for category presence in the cart
    category_flags = pd.get_dummies(df[['increment_id', 'category_name_1']], columns=['category_name_1'])
    category_flags = category_flags.groupby('increment_id').sum().reset_index()
    cart_features = pd.merge(cart_features, category_flags, on='increment_id', how='left')

    # Step 7: One-hot encode payment methods
    payment_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    payment_encoded = payment_encoder.fit_transform(cart_features[['payment_method']])
    payment_encoded_df = pd.DataFrame(payment_encoded, columns=payment_encoder.get_feature_names_out(['payment_method']))
    cart_features = pd.concat([cart_features, payment_encoded_df], axis=1)
    cart_features.drop(columns=['payment_method'], inplace=True)

    # Step 8: Add customer purchase history
    purchase_history = df.groupby('Customer ID').agg(
        customer_total_spent=('grand_total', 'sum'),
        customer_total_orders=('increment_id', 'nunique')
    ).reset_index()
    cart_features = pd.merge(cart_features, purchase_history, left_on='customer_id', right_on='Customer ID', how='left')
    cart_features.drop(columns=['Customer ID'], inplace=True)

    # Step 9: Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = ['total_price', 'total_items', 'total_discount', 'num_unique_categories', 'customer_total_spent', 'customer_total_orders']
    cart_features[numerical_features] = scaler.fit_transform(cart_features[numerical_features])

    # Step 10: Drop ID columns before training
    cart_features.drop(columns=['increment_id', 'customer_id'], inplace=True, errors='ignore')

    # Step 11: Handle missing values
    cart_features.fillna(0, inplace=True)

    # Step 12: Save the preprocessed dataset
    cart_features.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to {output_file}")


def visualize_preprocessed_data(preprocessed_file):
    # Load the preprocessed dataset
    df = pd.read_csv(preprocessed_file)

    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    fig.suptitle('Abandonment Analysis', fontsize=16, y=0.92)

    # 1. Total Price vs. Abandonment
    sns.boxplot(data=df, x='abandoned', y='total_price', hue='abandoned', palette='coolwarm', ax=axes[0, 0], dodge=False, legend=False)
    axes[0, 0].set_title('Total Price vs Abandonment')
    axes[0, 0].set_xlabel('Abandoned (0: Not Abandoned, 1: Abandoned)')
    axes[0, 0].set_ylabel('Total Price')

    # 2. Total Items vs. Abandonment
    sns.boxplot(data=df, x='abandoned', y='total_items', hue='abandoned', palette='coolwarm', ax=axes[0, 1], dodge=False, legend=False)
    axes[0, 1].set_title('Total Items vs Abandonment')
    axes[0, 1].set_xlabel('Abandoned (0: Not Abandoned, 1: Abandoned)')
    axes[0, 1].set_ylabel('Total Items')

    # 3. Abandonment Rate by Payment Method
    if 'abandoned' in df.columns and any(col.startswith('payment_method_') for col in df.columns):
        payment_methods = [col for col in df.columns if col.startswith('payment_method_')]
        payment_data = {method: df.groupby(method)['abandoned'].mean().iloc[1] for method in payment_methods}
        payment_data = pd.DataFrame(list(payment_data.items()), columns=['Payment Method', 'Abandonment Rate'])
        sns.barplot(x='Abandonment Rate', y='Payment Method', data=payment_data, ax=axes[1, 0])
        axes[1, 0].set_title('Abandonment Rate by Payment Method')
        axes[1, 0].set_xlabel('Abandonment Rate')
        axes[1, 0].set_ylabel('Payment Method')
    else:
        axes[1, 0].set_visible(False)

    # 4. Abandonment Rate by Categories
    if 'abandoned' in df.columns and any(col.startswith('category_name_1_') for col in df.columns):
        category_columns = [col for col in df.columns if col.startswith('category_name_1_')]
        category_data = {category: df.groupby(category)['abandoned'].mean().iloc[1] for category in category_columns}
        category_data = pd.DataFrame(list(category_data.items()), columns=['Category', 'Abandonment Rate'])
        sns.barplot(x='Abandonment Rate', y='Category', data=category_data, ax=axes[1, 1])
        axes[1, 1].set_title('Abandonment Rate by Categories')
        axes[1, 1].set_xlabel('Abandonment Rate')
        axes[1, 1].set_ylabel('Categories')
    else:
        axes[1, 1].set_visible(False)

    # 5. Total Orders vs. Abandonment
    sns.boxplot(data=df, x='abandoned', y='customer_total_orders', hue='abandoned', palette='coolwarm', ax=axes[2, 0], dodge=False, legend=False)
    axes[2, 0].set_title('Customer Total Orders vs Abandonment')
    axes[2, 0].set_xlabel('Abandoned (0: Not Abandoned, 1: Abandoned)')
    axes[2, 0].set_ylabel('Total Orders')

    # 6. Total Spending vs. Abandonment
    sns.boxplot(data=df, x='abandoned', y='customer_total_spent', hue='abandoned', palette='coolwarm', ax=axes[2, 1], dodge=False, legend=False)
    axes[2, 1].set_title('Customer Total Spending vs Abandonment')
    axes[2, 1].set_xlabel('Abandoned (0: Not Abandoned, 1: Abandoned)')
    axes[2, 1].set_ylabel('Total Spending')

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