# -*- coding: utf-8 -*-
"""
Kaggle submission - House Prices: Advanced Regression Techniques
Uses the same preprocessing and the optimised Random Forest from the assignment,
trained on the full train.csv, then predicts SalePrice for test.csv.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
test_ids = test_df['Id']

# Remove the two outliers (training data only - every test house needs a prediction)
train_df = train_df[~train_df['Id'].isin([1299, 524])]

X_train = train_df.drop(columns=['Id', 'SalePrice'])
y_train = train_df['SalePrice']
X_test = test_df.drop(columns=['Id'])

print('Training set:', X_train.shape)
print('Test set    :', X_test.shape)


# Handle missing values
def handle_missing_data(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Same steps as the assignment
    drop_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt']
    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    X_train['MasVnrArea'] = X_train['MasVnrArea'].fillna(0)
    X_test['MasVnrArea'] = X_test['MasVnrArea'].fillna(0)

    neighborhood_median = X_train.groupby('Neighborhood')['LotFrontage'].median()
    X_train['LotFrontage'] = X_train['LotFrontage'].fillna(X_train['Neighborhood'].map(neighborhood_median))
    X_test['LotFrontage'] = X_test['LotFrontage'].fillna(X_test['Neighborhood'].map(neighborhood_median))

    # New for test.csv: missing basement/garage numbers mean no basement/garage
    zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
    for col in zero_cols:
        X_train[col] = X_train[col].fillna(0)
        X_test[col] = X_test[col].fillna(0)

    # New for test.csv: fill remaining categories with the training-set mode
    mode_cols = ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st',
                 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
    for col in mode_cols:
        mode_value = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_value)
        X_test[col] = X_test[col].fillna(mode_value)

    none_cols = ['MasVnrType', 'FireplaceQu', 'GarageType',
                 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in none_cols:
        X_train[col] = X_train[col].fillna('None')
        X_test[col] = X_test[col].fillna('None')

    bsmt_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
    for col in bsmt_cols:
        train_no_bsmt = X_train['TotalBsmtSF'] == 0
        test_no_bsmt = X_test['TotalBsmtSF'] == 0
        X_train.loc[train_no_bsmt, col] = X_train.loc[train_no_bsmt, col].fillna('None')
        X_test.loc[test_no_bsmt, col] = X_test.loc[test_no_bsmt, col].fillna('None')

        mode_value = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_value)
        X_test[col] = X_test[col].fillna(mode_value)

    print('\nMissing values after handling:')
    print('Train:', X_train.isnull().sum().sum())
    print('Test :', X_test.isnull().sum().sum())

    return X_train, X_test


# Categorical encoding (same as the assignment)
def cat_encoding(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cat = ['MSSubClass', 'MoSold', 'YrSold']
    X_train[num_cat] = X_train[num_cat].astype(str)
    X_test[num_cat] = X_test[num_cat].astype(str)

    qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    fintype_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

    ordinal_maps = {
        'ExterQual': qual_map, 'ExterCond': qual_map,
        'BsmtQual': qual_map, 'BsmtCond': qual_map,
        'HeatingQC': qual_map, 'KitchenQual': qual_map,
        'FireplaceQu': qual_map, 'GarageQual': qual_map,
        'GarageCond': qual_map,
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'BsmtFinType1': fintype_map,
        'BsmtFinType2': fintype_map,
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
        'Functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3,
                       'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
    }
    for col, mapping in ordinal_maps.items():
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)

    binary_maps = {'CentralAir': {'N': 0, 'Y': 1}, 'Street': {'Grvl': 0, 'Pave': 1}}
    for col, mapping in binary_maps.items():
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)

    nominal_cols = X_train.select_dtypes(exclude='number').columns
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    train_ohe = ohe.fit_transform(X_train[nominal_cols])
    test_ohe = ohe.transform(X_test[nominal_cols])
    columns = ohe.get_feature_names_out(nominal_cols)

    train_ohe = pd.DataFrame(train_ohe, columns=columns, index=X_train.index)
    test_ohe = pd.DataFrame(test_ohe, columns=columns, index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=nominal_cols), train_ohe], axis=1)
    X_test = pd.concat([X_test.drop(columns=nominal_cols), test_ohe], axis=1)

    print('\nAfter encoding - Train:', X_train.shape, 'Test:', X_test.shape)
    print('Missing values after encoding:',
          X_train.isnull().sum().sum(), '(train),', X_test.isnull().sum().sum(), '(test)')

    return X_train, X_test


# Log transformation and scaling (same as the assignment)
def transform_and_scale(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    continuous = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea',
        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
        'ScreenPorch', 'PoolArea', 'MiscVal'
    ]
    skewness = X_train[continuous].skew()
    candidates = skewness[skewness.abs() > 0.75].index
    after = np.log1p(X_train[candidates]).skew()
    skewed_cols = after[after.abs() < skewness[candidates].abs()].index

    X_train[skewed_cols] = np.log1p(X_train[skewed_cols])
    X_test[skewed_cols] = np.log1p(X_test[skewed_cols])

    binary_cols = [col for col in X_train.columns if X_train[col].isin([0, 1]).all()]
    scale_cols = X_train.columns.difference(binary_cols)

    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    print('\nLog-transformed features:', len(skewed_cols))
    print('Scaled features:', len(scale_cols), '| Binary features:', len(binary_cols))

    return X_train, X_test


X_train, X_test = handle_missing_data(X_train, X_test)
X_train, X_test = cat_encoding(X_train, X_test)
X_train_s, X_test_s = transform_and_scale(X_train, X_test)
y_train_log = np.log1p(y_train)


# Optimised Random Forest (best parameters from the assignment's RandomizedSearchCV)
final_model = Pipeline([
    ('select', SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        threshold=-np.inf,
        max_features=150
    )),
    ('rf', RandomForestRegressor(
        n_estimators=500,
        max_features=0.25,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    ))
])

# Expected Kaggle score: 5-fold CV RMSE on log prices (same metric Kaggle uses)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = -cross_val_score(final_model, X_train_s, y_train_log,
                             cv=kf, scoring='neg_root_mean_squared_error')
print(f'\n5-fold CV RMSE (log): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

# Train on the full training data and predict test.csv
final_model.fit(X_train_s, y_train_log)
test_pred = np.expm1(final_model.predict(X_test_s))

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_pred})
os.makedirs('submission', exist_ok=True)
submission.to_csv('submission/submission.csv', index=False)

# Final checks before uploading
print('\nSubmission saved: submission/submission.csv')
print('Rows:', len(submission), '| Missing:', submission['SalePrice'].isnull().sum())
print(submission['SalePrice'].describe().round(0))
print(submission.head())
