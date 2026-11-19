# -*- coding: utf-8 -*-
"""
MAI5043 Applied Machine Learning - Regression Assignment
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, zscore
from sklearn.model_selection import train_test_split,RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings('ignore')

'''
Task 1 & 2: Load data and perform EDA
'''
#Load data
train_df = pd.read_csv('train.csv')
pd.set_option("display.max_columns", None)  
pd.set_option("display.max_rows", None)      

#Dataset overview
def dataset_summary():
    print("Size for train dataset (rows, columns):", train_df.shape)
    print(train_df.head())
    
    train_df.info()
    
    print("\nSummary statistics features:")
    print(train_df.describe().T.round(1))

dataset_summary()

plot_counter = {}

def save_plot(name, folder='figures'):
       os.makedirs(folder, exist_ok=True)
       plot_counter[name] = plot_counter.get(name, 0) + 1
       filepath = os.path.join(folder, f'{name}_{plot_counter[name]}.png')
       plt.savefig(filepath, dpi=300, bbox_inches='tight')
       print(f'Saved: {filepath}')

#SalePrice histogram with fitted normal curve, and Quantile–Quantile (Q-Q) plot
def plot_distribution(data, name):
    fig, ax = plt.subplots(2, 1, figsize=(7, 10))

    mu, sigma = norm.fit(data)

    sns.histplot(data, kde=True, stat='density', ax=ax[0],
                 label='Actual distribution (KDE)')

    x = np.linspace(data.min(), data.max(), 200)
    ax[0].plot(x, norm.pdf(x, mu, sigma), color='black',label=f'Normal Distribution (μ={mu:.2f}, σ={sigma:.2f})')

    ax[0].set_title(f'{name} (Skewness={data.skew():.2f})')
    ax[0].legend()

    # Q-Q plot
    stats.probplot(data, plot=ax[1])
    ax[1].set_title(f'Q-Q Plot of {name}')

    plt.tight_layout()

    save_plot(name.replace(' ', '_').lower())
    plt.show()
    plt.close()

# Original SalePrice
plot_distribution(train_df['SalePrice'], 'SalePrice')

# Log-transformed SalePrice
plot_distribution(np.log1p(train_df['SalePrice']), 'Log SalePrice')

#Scatter Plot
def scatter_plot():

    num_features = train_df.select_dtypes(include='number').columns.drop(
        ['Id', 'SalePrice', 'MSSubClass', 'MoSold', 'YrSold']
    )

    for i in range(0, len(num_features), 6):

        features = num_features[i:i+6]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for ax, col in zip(axes.flat, features):

            sns.regplot(
                data=train_df,
                x=col,
                y='SalePrice',
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'},
                ci=None,
                ax=ax
            )

            ax.set_xlabel(col)
            ax.set_ylabel('SalePrice')
            ax.set_title(f'{col} vs SalePrice')

        for ax in axes.flat[len(features):]:
            ax.remove()

        plt.tight_layout()

        save_plot('scatterplot')

        plt.show()
        plt.close()


scatter_plot()

def histogram_plot():

    num_features = train_df.select_dtypes(include='number').columns.drop(['Id', 'SalePrice'])

    discrete_features = [
        'MSSubClass', 'MoSold', 'YrSold',
        'OverallQual', 'OverallCond',
        'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr',
        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars'
    ]

    num_features = num_features.drop(
        [col for col in discrete_features if col in num_features]
    )

    for i in range(0, len(num_features), 6):

        features = num_features[i:i+6]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for ax, col in zip(axes.flat, features):
            sns.histplot(train_df[col], kde=True, ax=ax)
            ax.set_title(f'{col} Distribution')

        for ax in axes.flat[len(features):]:
            ax.remove()

        plt.tight_layout()

        save_plot('histogram')

        plt.show()
        plt.close()

histogram_plot()

def box_plot():

    cat_features = train_df.select_dtypes(include='object').columns.tolist()
    cat_features += [
       'MSSubClass', 'MoSold', 'YrSold',
       'OverallQual', 'OverallCond',
       'BsmtFullBath', 'BsmtHalfBath',
       'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars'
       ]
    for i in range(0, len(cat_features), 6):

        features = cat_features[i:i+6]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for ax, col in zip(axes.flat, features):

            sns.boxplot(
                x=train_df[col].fillna('Missing/NA').astype(str),
                y=train_df['SalePrice'],
                ax=ax
            )

            ax.set_title(f'SalePrice by {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('SalePrice')
            ax.tick_params(axis='x', rotation=45)

        for ax in axes.flat[len(features):]:
            ax.remove()

        plt.tight_layout()

        save_plot('boxplot')

        plt.show()
        plt.close()

box_plot()

def count_plot():

    cat_features = train_df.select_dtypes(include='object').columns.tolist()

    cat_features += [
        'MSSubClass', 'MoSold', 'YrSold',
        'OverallQual', 'OverallCond',
        'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr',
        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars'
    ]

    for i in range(0, len(cat_features), 6):

        features = cat_features[i:i+6]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for ax, col in zip(axes.flat, features):

            data = train_df[col].fillna('Missing/NA').astype(str)

            sns.countplot(
                x=data,
                ax=ax,
                order=data.value_counts().index
            )

            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)

        for ax in axes.flat[len(features):]:
            ax.remove()

        plt.tight_layout()

        save_plot('countplot')

        plt.show()
        plt.close()

count_plot()

def correlation_heatmap():

    num_features = train_df.select_dtypes(include='number').drop(columns=['Id'])
    corr = num_features.corr()

    plt.figure(figsize=(18, 14))

    sns.heatmap(
        corr,
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        linewidths=0.5
    )

    plt.title('Correlation Heatmap')
    plt.tight_layout()

    save_plot('correlation_heatmap')

    plt.show()
    plt.close()

    return corr

correlation_heatmap()

def saleprice_correlation_heatmap():

    num_features = train_df.select_dtypes(include='number').drop(columns=['Id'])

    corr = num_features.corr()

    saleprice_corr = corr['SalePrice'].drop('SalePrice')
    selected_features = saleprice_corr[abs(saleprice_corr) >= 0.5].index.tolist()

    selected_features.append('SalePrice')

    filtered_corr = corr.loc[selected_features, selected_features]

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        filtered_corr,
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        linewidths=0.5
    )

    plt.title('Correlation Heatmap (Features with |SalePrice Correlation| >= 0.5)')
    plt.tight_layout()

    save_plot('saleprice_correlation_heatmap')

    plt.show()
    plt.close()

saleprice_correlation_heatmap()

def show_missing_data(data, title, plot=True):

    missing_data = pd.DataFrame({
        'Total': data.isnull().sum(),
        'Percent': data.isnull().mean() * 100
    })

    missing_data = missing_data[missing_data['Total'] > 0]
    missing_data = missing_data.sort_values(by='Total', ascending=False)

    print(f'\n{title}')
    print(f'Total features: {data.shape[1]}')
    print(f'Features with missing values: {len(missing_data)}')
    print(f'Total missing values: {data.isnull().sum().sum()}')

    if missing_data.empty:
        print("No missing values found.")
    else:
        print(missing_data.round(2))

    if plot and not missing_data.empty:
        plt.figure(figsize=(12, 6))

        sns.barplot(
            x=missing_data.index,
            y=missing_data['Percent'],
            palette='YlGnBu'
        )

        plt.xticks(rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Missing Values (%)')
        plt.title(title)
        plt.tight_layout()

        save_plot('missing_values')

        plt.show()
        plt.close()

show_missing_data(train_df,"Train Dataset - Percentage of Missing Values",plot=True)

'''
Task 3: Data Preprocessing
'''
# Handle duplicates
def dup_check():
    columns = train_df.columns.drop('Id')
    
    before = train_df.duplicated(subset=columns).sum()
    train_df.drop_duplicates(subset=columns, inplace=True)
    after = train_df.duplicated(subset=columns).sum()

    print("Duplicate before process:", before)
    print("Duplicates after process:", after)

dup_check()

#Handle Outlier
def check_outliers():

    # Selected features for outlier detection
    num_features = [
        'YearBuilt',
        'YearRemodAdd',
        'TotalBsmtSF',
        '1stFlrSF',
        'GrLivArea',
        'GarageArea',
    ]

    outlier_count = pd.Series(0, index=train_df.index)

    for col in num_features:

        # IQR method
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        iqr_outliers = (train_df[col] < lower) | (train_df[col] > upper)

        # Z-score method
        z_scores = zscore(train_df[col].fillna(train_df[col].mean()))
        z_outliers = abs(z_scores) > 3

        outlier_count += iqr_outliers.astype(int)
        outlier_count += z_outliers.astype(int)

    results = train_df[['Id'] + num_features + ['OverallQual','SalePrice','SaleCondition']].copy()

    results['Outlier_Count'] = outlier_count
    
    results = results[results['Outlier_Count'] > 0]
    results = results.sort_values('Outlier_Count', ascending=False)
    
    print('\nHouses flagged by IQR / Z-score (top 15):')
    print(results.head(15).to_string(index=False))

check_outliers()

def drop_outlier():

    id_no = [1299, 524]

    global train_df
    before = len(train_df)
    train_df = train_df[~train_df['Id'].isin(id_no)]

    print("\nOriginal dataset:", before)
    print("After removing outliers:", len(train_df))
    print("Rows removed:", before - len(train_df))

drop_outlier()

# Handle missing data
def split_train_test_data():
    X = train_df.drop(columns=['Id', 'SalePrice'])
    y = train_df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle= True)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test_data()

def handle_missing_data():
    show_missing_data(X_train, "Before Preprocessing", plot=False)
    # 1. Drop columns
    drop_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence','GarageYrBlt']

    X_train.drop(columns=drop_cols, inplace=True)
    X_test.drop(columns=drop_cols, inplace=True)
    
    X_train['MasVnrArea'] = X_train['MasVnrArea'].fillna(0)
    X_test['MasVnrArea'] = X_test['MasVnrArea'].fillna(0)
    
    neighborhood_median = X_train.groupby('Neighborhood')['LotFrontage'].median()

    X_train['LotFrontage'] = X_train['LotFrontage'].fillna(X_train['Neighborhood'].map(neighborhood_median))
    X_test['LotFrontage'] = X_test['LotFrontage'].fillna(X_test['Neighborhood'].map(neighborhood_median))
    
    electrical_mode = X_train['Electrical'].mode()[0]
    
    X_train['Electrical'] = X_train['Electrical'].fillna(electrical_mode)
    X_test['Electrical'] = X_test['Electrical'].fillna(electrical_mode)
    
    # 2. Fill with 'None'
    none_cols = [
        'MasVnrType',
        'FireplaceQu', 
        'GarageType', 
        'GarageFinish',
        'GarageQual',
        'GarageCond'
    ]

    for col in none_cols:
        X_train[col] = X_train[col].fillna('None')
        X_test[col] = X_test[col].fillna('None')


    bsmt_cols = [
        'BsmtExposure',
        'BsmtFinType2',
        'BsmtQual',
        'BsmtCond',
        'BsmtFinType1'
    ]

    for col in bsmt_cols:
        train_no_bsmt = X_train['TotalBsmtSF'] == 0
        test_no_bsmt = X_test['TotalBsmtSF'] == 0

        X_train.loc[train_no_bsmt, col] = X_train.loc[train_no_bsmt, col].fillna('None')
        X_test.loc[test_no_bsmt, col] = X_test.loc[test_no_bsmt, col].fillna('None')

        # Handle any remaining missing values using the training mode
        mode_value = X_train[col].mode()[0]
        
        X_train[col] = X_train[col].fillna(mode_value)
        X_test[col] = X_test[col].fillna(mode_value)
        
    show_missing_data(X_train, "After Preprocessing", plot=False)
        
handle_missing_data()

# Categorical encoding
def cat_encoding(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Convert numerical categories into strings
    num_cat = ['MSSubClass', 'MoSold', 'YrSold']
    X_train[num_cat] = X_train[num_cat].astype(str)
    X_test[num_cat] = X_test[num_cat].astype(str)

    # Ordinal encoding
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

    # Binary encoding
    binary_maps = {
        'CentralAir': {'N': 0, 'Y': 1},
        'Street': {'Grvl': 0, 'Pave': 1}
    }

    for col, mapping in binary_maps.items():
        X_train[col] = X_train[col].map(mapping)
        X_test[col] = X_test[col].map(mapping)

    # One-hot encoding
    nominal_cols = X_train.select_dtypes(exclude='number').columns

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    train_ohe = ohe.fit_transform(X_train[nominal_cols])
    test_ohe = ohe.transform(X_test[nominal_cols])

    columns = ohe.get_feature_names_out(nominal_cols)

    train_ohe = pd.DataFrame(train_ohe, columns=columns, index=X_train.index)
    test_ohe = pd.DataFrame(test_ohe, columns=columns, index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=nominal_cols), train_ohe], axis=1)
    X_test = pd.concat([X_test.drop(columns=nominal_cols), test_ohe], axis=1)

    print("\nDataset shape after categorical encoding:")
    print("Training set:", X_train.shape)
    print("Testing set:", X_test.shape)
    print("Remaining categorical columns:", X_train.select_dtypes(exclude='number').columns.tolist())

    return X_train, X_test

X_train, X_test = cat_encoding(X_train, X_test)


def transform_and_scale(X_train, X_test, y_train, y_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

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

    # Standardise numerical features, keeping binary columns unchanged
    binary_cols = [
        col for col in X_train.columns
        if X_train[col].isin([0, 1]).all()
    ]

    scale_cols = X_train.columns.difference(binary_cols)

    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    
    print("\nLog-transformed features:", list(skewed_cols))
    print("Number of scaled features:", len(scale_cols))
    print("Number of binary features:", len(binary_cols))
    
    print("\nMissing values:")
    print("Train:", X_train.isna().sum().sum())
    print("Test:", X_test.isna().sum().sum())
    
    print("\nScaled training data:")
    print("Mean:", round(X_train[scale_cols].mean().mean(), 3))
    print("Standard deviation:", round(X_train[scale_cols].std().mean(), 3))

    return X_train, X_test, y_train, y_test

X_train_s, X_test_s, y_train_log, y_test_log = transform_and_scale(X_train, X_test, y_train, y_test)

'''
Task 4 - Train, test, and evaluate the Model
'''
#Train the model
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_model.fit(X_train_s, y_train_log)
    
#Evaluate the model
def evaluate_model(model, X_train_data, X_test_data, model_name='model', show_plots=True, return_metrics=False):

    train_pred = np.expm1(model.predict(X_train_data))
    test_pred = np.expm1(model.predict(X_test_data))

    results = []

    for name, actual, predicted in [
        ('Train', y_train, train_pred),
        ('Test', y_test, test_pred)
    ]:
        mse = mean_squared_error(actual, predicted)

        results.append({
            'Set': name,
            'MAE': mean_absolute_error(actual, predicted),
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'R2': r2_score(actual, predicted)
        })

        print(f'\n{name} Results')
        print('MAE :', round(results[-1]['MAE'], 2))
        print('MSE :', round(mse, 2))
        print('RMSE:', round(np.sqrt(mse), 2))
        print('R2  :', round(results[-1]['R2'], 4))

    if show_plots:
        residuals = y_test - test_pred

        # Actual vs Predicted Plot
        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, test_pred, alpha=0.5)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color='red'
        )
        plt.xlabel('Actual SalePrice ($)')
        plt.ylabel('Predicted SalePrice ($)')
        plt.title('Actual vs Predicted House Prices')
        plt.tight_layout()

        save_plot(f'{model_name}_actual_vs_predicted')

        plt.show()
        plt.close()

        # Residual Plot
        plt.figure(figsize=(7, 5))
        plt.scatter(test_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='red')
        plt.xlabel('Predicted SalePrice ($)')
        plt.ylabel('Residuals ($)')
        plt.title('Residual Plot')
        plt.tight_layout()

        save_plot(f'{model_name}_residual_plot')

        plt.show()
        plt.close()

    if return_metrics:
        return pd.DataFrame(results)

    return test_pred


original_results = evaluate_model(rf_model, X_train_s, X_test_s,model_name='original_rf')

def plot_feature_importance(model, top_n=15):
    importance = pd.Series(
        model.feature_importances_,
        index=X_train_s.columns
    ).sort_values(ascending=False).head(top_n)
    
    print(f'\nTop {top_n} Feature Importance:')
    print(importance.round(4).to_string())

    importance.sort_values().plot(kind='barh', figsize=(9, 6))

    plt.title(f'Top {top_n} Feature Importance')  
    plt.xlabel('Importance')
    plt.tight_layout()

    save_plot('feature_importance')               

    plt.show()
    plt.close()

plot_feature_importance(rf_model)

'''
Task 6 - Optimise the model
'''
# Create Pipeline
rf_pipeline = Pipeline([
    ('select', SelectFromModel(
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        ),
        threshold=-np.inf
    )),
    ('rf', RandomForestRegressor(
        random_state=42,
        n_jobs=1
    ))
])


# Define parameters for tuning
param_distributions = {
    'select__max_features': [30, 50, 60, 75, 100, 150, X_train.shape[1]],
    'rf__n_estimators': [300, 500],
    'rf__max_features': [0.2, 0.25, 0.33, 0.5],
    'rf__max_depth': [15, 20, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__bootstrap': [True, False]
}


# Perform Randomized Search
def optimise_model():

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_distributions,
        n_iter=40,
        scoring='neg_root_mean_squared_error',
        cv=kf,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train_s, y_train_log)

    print("\nBest CV RMSE:", round(-search.best_score_, 4))
    print("Best Parameters:", search.best_params_)

    return search.best_estimator_

best_model = optimise_model()


# Compare original and optimised models
print("\nOriginal Random Forest")
original_results = evaluate_model(rf_model,X_train_s,X_test_s,show_plots=False,return_metrics=True)

print("\nOptimised Random Forest")
optimised_results = evaluate_model(best_model,X_train_s,X_test_s,show_plots=False,return_metrics=True)


original_results.insert(0, 'Model', 'Original RF')
optimised_results.insert(0, 'Model', 'Optimised RF')

comparison = pd.concat(
    [original_results, optimised_results],
    ignore_index=True
    )

print("\nOriginal vs Optimised Model")
print(comparison.to_string(index=False, formatters={
    'MAE': '{:,.2f}'.format,
    'MSE': '{:,.2f}'.format,
    'RMSE': '{:,.2f}'.format,
    'R2': '{:.4f}'.format
    }))


# Calculate test performance improvement
test = comparison[comparison['Set'] == 'Test'].set_index('Model')

for metric in ['MAE','MSE','RMSE']:
    change = (
        (test.loc['Optimised RF', metric] - test.loc['Original RF', metric])/ test.loc['Original RF', metric] * 100 )
    print(f'Test {metric} change: {change:.2f}%')

print(
    'Test R2 change:',round(test.loc['Optimised RF', 'R2'] - test.loc['Original RF', 'R2'], 4)
)


# Check selected features
n_selected = best_model.named_steps['select'].get_support().sum()

print("\nNumber of features:")
print("Original:", X_train.shape[1])
print("Selected:", n_selected)


# Evaluate optimised model with plots
y_pred_optimised = evaluate_model(best_model, X_train_s, X_test_s,model_name='optimised_rf')
