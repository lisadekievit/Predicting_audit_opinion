import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def remove_outliers_iqr(X, y, columns, min_outliers=8):
    """
    Removes rows from X and y where a row has outliers in at least `min_outliers` columns 
    using the IQR method. Updates y accordingly to ensure alignment, and returns 
    how many rows were deleted, broken down by y class.
    
    Parameters:
    X (DataFrame): Input DataFrame for features
    y (Series or DataFrame): Target variable that needs to stay aligned with X
    columns (list): List of column names where outliers need to be removed
    min_outliers (int): Minimum number of outliers in a row before it gets deleted
    
    Returns:
    Tuple: (X_no_outliers, y_no_outliers, removed_counts)
           - DataFrames with outliers removed, and a dictionary with counts of removed rows by y class.
    """
    X_out = X.copy()
    y_out = y.copy()

    removed_counts = {}  # Dictionary to store count of removed rows per class

    # Create a DataFrame to track the number of outliers per row
    outlier_mask = pd.DataFrame(False, index=X.index, columns=columns)

    for col in columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = X_out[col].quantile(0.25)
        Q3 = X_out[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for detecting outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Mark outliers in the column
        outlier_mask[col] = (X_out[col] < lower_bound) | (X_out[col] > upper_bound)
    
    # Count the number of outlier columns per row
    outliers_per_row = outlier_mask.sum(axis=1)
    
    # Keep only rows with fewer than `min_outliers` outliers
    full_mask = outliers_per_row < min_outliers
    
    # Apply the final mask to both X and y
    X_out = X_out[full_mask]
    y_out = y_out[full_mask]
    
    # Calculate how many rows were removed
    removed_mask = ~full_mask
    removed_y = y[removed_mask]
    
    # Count the number of removed rows per class
    removed_counts = removed_y.value_counts().to_dict()
    
    return X_out, y_out, removed_counts

def preprocess_and_apply_smote1(X_train, y_train, max_missing=6, imputation_strategy='mean'):
    """
    Removes rows with more than `max_missing` missing values, imputes the rest, and applies SMOTE.
    
    Parameters:
    X_train (DataFrame): Training feature set
    y_train (Series): Training labels
    max_missing (int): Maximum number of missing values allowed per row before removal
    imputation_strategy (str): Strategy for imputing missing values ('mean', 'median', etc.)
    
    Returns:
    Tuple: (X_train_smote_df, y_train_smote)
           - X_train_smote_df is a DataFrame after SMOTE with original column names
           - y_train_smote is a Series with the resampled target labels
    """

    # Step 1: Remove rows with 6 or more missing values
    X_train_cleaned = X_train[X_train.isnull().sum(axis=1) < max_missing]
    y_train_cleaned = y_train.loc[X_train_cleaned.index]

    # Step 2: Impute the remaining missing values
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_train_imputed = imputer.fit_transform(X_train_cleaned)  # Returns a NumPy array

    # Step 3: Apply SMOTE to the cleaned and imputed dataset
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_imputed, y_train_cleaned)

    # Convert the NumPy array back to DataFrame with original column names
    X_train_smote_df = pd.DataFrame(X_train_smote, columns=X_train_cleaned.columns)

    return X_train_smote_df, y_train_smote
"""X_train, y_train = k.preprocess_and_apply_smote1(X_train, y_train, max_missing=6, imputation_strategy='mean')
print("Class distribution after SMOTE:", pd.Series(y_train).value_counts())"""

def scale_columns2(df_train, df_test, columns_to_scale, imputation_strategy='mean'):
    # Step 1: Check and replace infinite values with NaN
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Step 2: Initialize the imputer with the specified strategy
    imputer = SimpleImputer(strategy=imputation_strategy)

    # Impute NaN values in the specified columns
    df_train[columns_to_scale] = imputer.fit_transform(df_train[columns_to_scale])
    df_test[columns_to_scale] = imputer.transform(df_test[columns_to_scale])

    # Step 3: Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the training set for specified columns
    df_train_scaled = df_train.copy()
    df_train_scaled[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])

    # Transform the validation and test sets using the same scaler
    df_test_scaled = df_test.copy()
    df_test_scaled[columns_to_scale] = scaler.transform(df_test[columns_to_scale])

    return df_train_scaled, df_test_scaled


def scale_columns1(df_train, df_val, df_test, columns_to_scale, imputation_strategy='mean'):
    # Step 1: Check and replace infinite values with NaN
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Step 2: Initialize the imputer with the specified strategy
    imputer = SimpleImputer(strategy=imputation_strategy)

    # Impute NaN values in the specified columns
    df_train[columns_to_scale] = imputer.fit_transform(df_train[columns_to_scale])
    df_val[columns_to_scale] = imputer.transform(df_val[columns_to_scale])
    df_test[columns_to_scale] = imputer.transform(df_test[columns_to_scale])

    # Step 3: Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the training set for specified columns
    df_train_scaled = df_train.copy()
    df_train_scaled[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])

    # Transform the validation and test sets using the same scaler
    df_val_scaled = df_val.copy()
    df_val_scaled[columns_to_scale] = scaler.transform(df_val[columns_to_scale])
    df_test_scaled = df_test.copy()
    df_test_scaled[columns_to_scale] = scaler.transform(df_test[columns_to_scale])

    return df_train_scaled, df_val_scaled, df_test_scaled


def pca_on_train_and_transform_full_data(X_train, X_val, X_test, columns=None):
    # Step 1: Apply PCA on training data
    pca = PCA()
    pca.fit(X_train)

    # Step 2: Get eigenvalues (explained variance)
    eigenvalues = pca.explained_variance_

    # Step 3: Select components with eigenvalues > 1
    components_with_eigenvalues_gt_1 = np.where(eigenvalues > 1)[0]

    # Step 4: Transform training, validation, and test data using the selected components
    X_train_pca = pca.transform(X_train)[:, components_with_eigenvalues_gt_1]
    X_val_pca = pca.transform(X_val)[:, components_with_eigenvalues_gt_1]
    X_test_pca = pca.transform(X_test)[:, components_with_eigenvalues_gt_1]

    # Step 5: Concatenate the PCA-selected components with the original data (full data)
    X_train_full = np.hstack([X_train_pca, X_train])
    X_val_full = np.hstack([X_val_pca, X_val])
    X_test_full = np.hstack([X_test_pca, X_test])

    # Step 6: Convert to Pandas DataFrame for easier manipulation
    pca_columns = [f'PCA_{i+1}' for i in range(X_train_pca.shape[1])]
    
    if columns is not None:
        original_columns = columns
    else:
        original_columns = [f'Original_{i+1}' for i in range(X_train.shape[1])]

    full_columns = pca_columns + original_columns

    X_train_full = pd.DataFrame(X_train_full, columns=full_columns)
    X_val_full = pd.DataFrame(X_val_full, columns=full_columns)
    X_test_full = pd.DataFrame(X_test_full, columns=full_columns)

    # Step 6: Get explained variance ratio of the selected components
    explained_variance_selected = pca.explained_variance_ratio_[components_with_eigenvalues_gt_1]

    # Optional: Plot the cumulative explained variance
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Components')
    plt.show()

    return X_train_full, X_val_full, X_test_full, explained_variance_selected, eigenvalues

def create_pca_table(explained_variance, eigenvalues):
    cumulative_explained_variance = explained_variance.cumsum()
    # Create a DataFrame for the table
    pca_table = pd.DataFrame({
        'Principal Component': [f'PC_{i+1}' for i in range(len(explained_variance))],
        'Eigenvalue': eigenvalues[:len(explained_variance)],  # Limit to number of selected PCs
        'Explained Variance': explained_variance,
        'Cumulative Explained Variance': cumulative_explained_variance
    })
    
    return pca_table

print("runned!")