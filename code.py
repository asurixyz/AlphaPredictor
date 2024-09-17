import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load training and test data
train_data = pd.read_csv('/home/asuri/Downloads/train_data.csv')
test_data = pd.read_csv('/home/asuri/Downloads/final_test_data.csv')

# Feature Engineering
def create_features(df):
    # Mid-Price
    df['midprice_1'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    df['midprice_2'] = (df['bid_price_2'] + df['ask_price_2']) / 2
    df['midprice_3'] = (df['bid_price_3'] + df['ask_price_3']) / 2
    df['midprice_4'] = (df['bid_price_4'] + df['ask_price_4']) / 2
    df['midprice_5'] = (df['bid_price_5'] + df['ask_price_5']) / 2

    df['weighted_mid_price_1'] = (df['bid_price_1'] * df['ask_volume_1'] + df['ask_price_1'] * df['bid_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'])

    # Price Spread
    df['spread_1'] = df['ask_price_1'] - df['bid_price_1']
    df['spread_2'] = df['ask_price_2'] - df['bid_price_2']
    df['spread_3'] = df['ask_price_3'] - df['bid_price_3']
    df['spread_4'] = df['ask_price_4'] - df['bid_price_4']
    df['spread_5'] = df['ask_price_5'] - df['bid_price_5']

    
    # Weighted Spread
    df['weighted_spread_1'] = df['ask_volume_1']*df['ask_price_1'] - df['bid_price_1'] * df['bid_volume_1']
    df['weighted_spread_2'] = df['ask_volume_2']*df['ask_price_2'] - df['bid_price_2'] * df['bid_volume_2']
    df['weighted_spread_3'] = df['ask_volume_3']*df['ask_price_3'] - df['bid_price_3'] * df['bid_volume_3']
    df['weighted_spread_4'] = df['ask_volume_4']*df['ask_price_4'] - df['bid_price_4'] * df['bid_volume_4']
    df['weighted_spread_5'] = df['ask_volume_5']*df['ask_price_5'] - df['bid_price_5'] * df['bid_volume_5']

    # Normalised Spread
    df['norm_spread_1'] = (df['ask_price_1'] - df['bid_price_1']) / df['midprice_1']
    df['norm_spread_2'] = (df['ask_price_2'] - df['bid_price_2']) / df['midprice_2']
    df['norm_spread_3'] = (df['ask_price_3'] - df['bid_price_3']) / df['midprice_3']
    df['norm_spread_4'] = (df['ask_price_4'] - df['bid_price_4']) / df['midprice_4']
    df['norm_spread_5'] = (df['ask_price_5'] - df['bid_price_5']) / df['midprice_5']


    # Volume Imbalance
    df['volume_imbalance_1'] = df['bid_volume_1'] - df['ask_volume_1'] / (df['bid_volume_1'] + df['ask_volume_1'])
    df['volume_imbalance_2'] = df['bid_volume_2'] - df['ask_volume_2'] / (df['bid_volume_2'] + df['ask_volume_2'])
    df['volume_imbalance_3'] = df['bid_volume_3'] - df['ask_volume_3'] / (df['bid_volume_3'] + df['ask_volume_3'])
    df['volume_imbalance_4'] = df['bid_volume_4'] - df['ask_volume_4'] / (df['bid_volume_4'] + df['ask_volume_4'])
    df['volume_imbalance_5'] = df['bid_volume_5'] - df['ask_volume_5'] / (df['bid_volume_5'] + df['ask_volume_5'])

    df['bid_ask_imbalance'] = df[[f'bid_volume_{i}' for i in range(1, 6)]].sum(axis=1) / (df[[f'ask_volume_{i}' for i in range(1, 6)]].sum(axis=1) + df[[f'bid_volume_{i}' for i in range(1, 6)]].sum(axis=1))

    # Order Flow (difference in price levels over time)
    df['order_flow'] = (df['bid_price_1'] - df['ask_price_1']) + (df['bid_volume_1'] - df['ask_volume_1'])

    # Depth (Total Volume at Top 5 Levels)
    df['depth'] = (df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].sum(axis=1) +
                   df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].sum(axis=1))


        # Feature 1: Price Momentum
    df['price_momentum'] = (df['last_trade_price'] - df['midprice']) / df['midprice']
    
    # Feature 2: Order Imbalance
    df['order_imbalance'] = (df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].sum(axis=1) - 
                             df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].sum(axis=1)) / \
                            (df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].sum(axis=1) + 
                             df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].sum(axis=1))

    # Feature 3: Liquidity Consumption (Impact)
    df['liquidity_impact'] = (df['recent_buy_order_count'] + df['recent_sell_order_count']) / df['total_order_count']
    
    # Feature 4: Order Flow Imbalance
    df['order_flow_imbalance'] = ((df['bid_price_1'] - df['bid_price_2']) - (df['ask_price_1'] - df['ask_price_2'])) / \
                                  ((df['bid_price_1'] + df['ask_price_1']) / 2)
    
    # Feature 5: Cumulative Depth
    df['cumulative_depth'] = df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].sum(axis=1) + \
                             df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].sum(axis=1)
    
    # Feature 6: Bid-Ask Slope
    df['bid_ask_slope'] = (df['ask_price_5'] - df['ask_price_1']) / (df['bid_price_1'] - df['bid_price_5'])
    
    # Feature 7: Volatility Estimator (Microprice)
    df['microprice'] = (df['bid_price_1'] * df['ask_volume_1'] + df['ask_price_1'] * df['bid_volume_1']) / \
                       (df['bid_volume_1'] + df['ask_volume_1'])
    
    # Feature 10: Market Pressure Indicator
    df['market_pressure'] = (df['recent_buy_order_count'] - df['recent_sell_order_count']) / df['total_order_count']



    df['midprice_slope'] = (df['midprice_5'] - df['midprice_1']) / 5

    df['market_depth_ratio'] = (df[[f'bid_volume_{i}' for i in range(1, 6)]].sum(axis=1)) / (df[[f'ask_volume_{i}' for i in range(1, 6)]].sum(axis=1))

    df['spread_skew'] = df['spread_1'] / df['spread_5']

    df['weighted_price_depth'] =  ((df[[f'bid_price_{i}' for i in range(1, 6)]] * df[[f'bid_volume_{i}' for i in range(1, 6)]]).sum(axis=1) + (df[[f'ask_price_{i}' for i in range(1, 6)]] * df[[f'ask_volume_{i}' for i in range(1, 6)]]).sum(axis=1))  / ((df[[f'bid_volume_{i}' for i in range(1, 6)]]).sum(axis=1) + (df[[f'ask_volume_{i}' for i in range(1, 6)]]).sum(axis=1))

    
    n = 5
    
    # Extract bid and ask prices and volumes
    bid_prices = df[[f'bid_price_{i}' for i in range(1, n+1)]].values
    ask_prices = df[[f'ask_price_{i}' for i in range(1, n+1)]].values
    bid_volumes = df[[f'bid_volume_{i}' for i in range(1, n+1)]].values
    ask_volumes = df[[f'ask_volume_{i}' for i in range(1, n+1)]].values
    
    ### 1. Price differences between adjacent levels (for both ask and bid sides)
    
    # Ask price differences between adjacent levels
    ask_price_diff = np.diff(ask_prices, axis=1)
    # Add columns to df for ask price differences
    for i in range(1, n):
        df[f'ask_price_diff_{i}_{i+1}'] = ask_price_diff[:, i-1]
    
    # Bid price differences between adjacent levels
    bid_price_diff = np.diff(bid_prices, axis=1)
    # Add columns to df for bid price differences
    for i in range(1, n):
        df[f'bid_price_diff_{i}_{i+1}'] = bid_price_diff[:, i-1]
    
    ### 2. Price and volume means for each side of each level
    
    # Price means for ask side
    df['ask_price_mean'] = np.mean(ask_prices, axis=1)
    # Price means for bid side
    df['bid_price_mean'] = np.mean(bid_prices, axis=1)
    
    # Volume means for ask side
    df['ask_volume_mean'] = np.mean(ask_volumes, axis=1)
    # Volume means for bid side
    df['bid_volume_mean'] = np.mean(bid_volumes, axis=1)
    
    ### 3. Accumulated differences for each level (cumulative sums)
    
    # Accumulated differences for ask prices
    df['ask_price_cumsum'] = np.cumsum(ask_prices, axis=1)[:, -1]
    # Accumulated differences for bid prices
    df['bid_price_cumsum'] = np.cumsum(bid_prices, axis=1)[:, -1]

    df['accumulated_price_diff'] = df['ask_price_cumsum'] - df['bid_price_cumsum']
    
    # Accumulated differences for ask volumes
    df['ask_vol_cumsum'] = np.cumsum(ask_volumes, axis=1)[:, -1]
    # Accumulated differences for bid volumes
    df['bid_vol_cumsum'] = np.cumsum(bid_volumes, axis=1)[:, -1]

    df['accumulated_volume_diff'] = df['ask_vol_cumsum'] - df['bid_vol_cumsum']

    df['vwap_1_2'] = ((df['bid_price_1'] * df['bid_volume_1'] + df['ask_price_1'] * df['ask_volume_1']) + 
                  (df['bid_price_2'] * df['bid_volume_2'] + df['ask_price_2'] * df['ask_volume_2']) ) / \
                  (df['bid_volume_1'] + df['ask_volume_1'] + 
                   df['bid_volume_2'] + df['ask_volume_2'] )

    df['bid_volume_momentum'] = df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].diff(axis=1).sum(axis=1)
    df['ask_volume_momentum'] = df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].diff(axis=1).sum(axis=1)

    df['bid_side_pressure'] = df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].sum(axis=1)
    df['ask_side_pressure'] = df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].sum(axis=1)
    df['order_book_pressure'] = df['bid_side_pressure'] / (df['ask_side_pressure'] + 1e-6)  # Prevent division by zero
    

    df['bid_price_sensitivity'] = (df['bid_price_5'] - df['bid_price_1']) / df[['bid_volume_1', 'bid_volume_5']].sum(axis=1)
    df['ask_price_sensitivity'] = (df['ask_price_5'] - df['ask_price_1']) / df[['ask_volume_1', 'ask_volume_5']].sum(axis=1)
    
    
    # 2. Liquidity Weight
    df['liquidity_weight'] = (df['bid_volume_1'] + df['ask_volume_1']) / \
                             (df['bid_volume_2'] + df['ask_volume_2'] + 1e-9)
    
    # 3. Spread Relative to Depth
    df['spread_depth_ratio_1'] = (df['ask_price_1'] - df['bid_price_1']) / (df['bid_volume_1'] + df['ask_volume_1'] + 1e-9)
    df['spread_depth_ratio_2'] = (df['ask_price_2'] - df['bid_price_2']) / (df['bid_volume_2'] + df['ask_volume_2'] + 1e-9)
    
    # 4. Volume Concentration Ratio
    df['volume_concentration_1'] = df['bid_volume_1'] / (df['bid_volume_1'] + df['bid_volume_2'] + 1e-9)
    df['volume_concentration_2'] = df['ask_volume_1'] / (df['ask_volume_1'] + df['ask_volume_2'] + 1e-9)
    
    # 5. Market Skewness
    df['market_skew'] = ((df['bid_volume_1'] - df['ask_volume_1']) + 
                         (df['bid_volume_2'] - df['ask_volume_2'])) / \
                         ((df['bid_volume_1'] + df['ask_volume_1']) + 
                          (df['bid_volume_2'] + df['ask_volume_2']) + 1e-9)

    # 6. Depth Imbalance (Bid vs Ask)
    df['depth_imbalance'] = (df['bid_volume_1'] + df['bid_volume_2']) / \
                            (df['ask_volume_1'] + df['ask_volume_2'] + 1e-9)
    
    # 7. Volume-Weighted Price
    df['vwap_1'] = (df['bid_price_1'] * df['bid_volume_1'] + df['ask_price_1'] * df['ask_volume_1']) / \
                   (df['bid_volume_1'] + df['ask_volume_1'] + 1e-9)

    df['vwap_2'] = (df['bid_price_2'] * df['bid_volume_2'] + df['ask_price_2'] * df['ask_volume_2']) / \
                   (df['bid_volume_2'] + df['ask_volume_2'] + 1e-9)
    
    df['vwap_3'] = (df['bid_price_3'] * df['bid_volume_3'] + df['ask_price_3'] * df['ask_volume_3']) / \
                   (df['bid_volume_3'] + df['ask_volume_3'] + 1e-9)

    df['vwap_5'] = (df['bid_price_5'] * df['bid_volume_5'] + df['ask_price_5'] * df['ask_volume_5']) / \
                   (df['bid_volume_5'] + df['ask_volume_5'] + 1e-9)
    
    df['vwap_4'] = (df['bid_price_4'] * df['bid_volume_4'] + df['ask_price_4'] * df['ask_volume_4']) / \
                   (df['bid_volume_4'] + df['ask_volume_4'] + 1e-9)

    for level in range(1, 6):
        df[f'depth_imbalance_{level}'] = (df[f'bid_volume_{level}']) / (df[f'ask_volume_{level}'] + 1e-9)

    df['cumulative_depth_imbalance'] = sum(
        df[f'depth_imbalance_{level}'] for level in range(1, 6)
    )

    df['depth_imbalance_full'] = df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].sum(axis=1) / \
                             df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].sum(axis=1)

    df['rvbb'] = df['bid_volume_1'] / df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].sum(axis=1)
    df['rvba'] = df['ask_volume_1'] / df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].sum(axis=1)
    df['rvbf'] = df['rvba'] + df['rvbb']

    df['depth_ratio_bid'] = df['bid_volume_1'] / df[['bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'bid_volume_4', 'bid_volume_5']].sum(axis=1)
    df['depth_ratio_ask'] = df['ask_volume_1'] / df[['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'ask_volume_4', 'ask_volume_5']].sum(axis=1)


    
    
    # df['ofi'] = sum(
    #      ((df[f'bid_volume_{level}'].diff() - df[f'ask_volume_{level}'].diff()) + 
    #                           (df[f'bid_price_{level}'].diff() - df[f'ask_price_{level}'].diff()))
    #     for level in range(1, 6)
    # ).fillna(0)

    # for level in range(1, 6):
    #     train_data[f'ofi_{level}'] = (
    #         (train_data[f'bid_volume_{level}'].diff() - train_data[f'ask_volume_{level}'].diff()) *
    #         (train_data[f'bid_price_{level}'].diff() - train_data[f'ask_price_{level}'].diff())
    #     ).fillna(0)
        
    #     test_data[f'ofi_{level}'] = (
    #         (test_data[f'bid_volume_{level}'].diff() - test_data[f'ask_volume_{level}'].diff()) *
    #         (test_data[f'bid_price_{level}'].diff() - test_data[f'ask_price_{level}'].diff())
    #     ).fillna(0)
    
    return df


# Apply feature engineering on both train and test data
train_data = create_features(train_data)
test_data = create_features(test_data)

# Define features and target for the train set
features = [
             #'midprice_1', 'midprice_2', 'midprice_3', 'midprice_4', 'midprice_5',
            'midprice_1',
            'last_trade_price',
            'recent_buy_order_count','recent_sell_order_count',
            # 'total_order_count','net_open_interest_change',
            'ask_price_1', 'bid_price_1', 'ask_volume_1', 'bid_volume_1',
            'ask_price_2', 'bid_price_2', 'ask_volume_2', 'bid_volume_2',
            'ask_price_3', 'bid_price_3', 'ask_volume_3', 'bid_volume_3',
            'ask_price_4', 'bid_price_4', 'ask_volume_4', 'bid_volume_4',
            'ask_price_5', 'bid_price_5', 'ask_volume_5', 'bid_volume_5',
            'weighted_spread_1', 'weighted_spread_2','weighted_spread_3','weighted_spread_4','weighted_spread_5',
            'volume_imbalance_1', # V imp
            'volume_imbalance_2', # V imp
            'volume_imbalance_3', # V imp
            'volume_imbalance_4', # V imp
            'volume_imbalance_5', # V imp
            'order_imbalance', # imp
            'price_momentum', # V imp
            'market_pressure',
            'microprice',  # V Imp
            'bid_ask_slope', # V Imp
            'liquidity_impact',
             'order_flow_imbalance',
            #'cumulative_depth',
            'accumulated_volume_diff',
            'accumulated_price_diff',
            'bid_vol_cumsum',
            'ask_vol_cumsum',
            'ask_price_mean', 'bid_price_mean',
            'ask_volume_mean', 'bid_volume_mean',
            'weighted_price_depth',
             'spread_skew',
            'market_depth_ratio',
            'midprice_slope',
            'bid_price_diff_1_2', 'bid_price_diff_2_3', 'bid_price_diff_3_4', 'bid_price_diff_4_5',
             'ask_price_diff_1_2', 'ask_price_diff_2_3', 'ask_price_diff_3_4', 'ask_price_diff_4_5',
            'weighted_mid_price_1',
            'depth',
             #'ofi_1', 'ofi_2', 'ofi_3', 'ofi_4', 'ofi_5',
             #'ofi',
            'bid_ask_imbalance',
            #'vwap_1_2',
            'bid_volume_momentum',
            'ask_volume_momentum',
            #'order_book_pressure',
            'bid_price_sensitivity',
            'ask_price_sensitivity',
            'depth_imbalance', # v imp,
            'depth_imbalance_1', # v imp,
            'depth_imbalance_2', # v imp,
            'depth_imbalance_3', # v imp,
            'depth_imbalance_4', # v imp,
            'depth_imbalance_5', # v imp,
             #'cumulative_depth_imbalance',
            'market_skew',
            'liquidity_weight',
            'spread_depth_ratio_1',
            'volume_concentration_1',
            'vwap_1','vwap_2', 'vwap_3','vwap_4','vwap_5',
            #'depth_imbalance_full',
            'rvbb', 'rvba','rvbf'
           ]

# Prepare the data
X = train_data[features]
y = train_data['actual_returns']

# Fill NaN values
X = X.fillna(method='ffill').fillna(method='bfill')
y = y.fillna(method='ffill').fillna(method='bfill')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train a Ridge regression model with cross-validation
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge = Ridge(random_state=42)
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on validation set
y_val_pred = best_model.predict(X_val)

# Calculate correlation
correlation = np.corrcoef(y_val, y_val_pred)[0, 1]
print(f"Validation Correlation: {correlation}")
print(f"Best alpha: {grid_search.best_params_['alpha']}")

# Predict on test data
test_features = test_data[features].fillna(method='ffill').fillna(method='bfill')
test_features_scaled = scaler.transform(test_features)
test_predictions = best_model.predict(test_features_scaled)

# Create submission file
submission = pd.DataFrame({
    'timestamp_code': test_data['timestamp_code'],
    'predicted_returns': test_predictions
})

# Ensure we have exactly 104980 rows
assert len(submission) == 104980, f"Submission has {len(submission)} rows instead of 104980"

submission.to_csv('submission.csv', index=False)
print(f"Submission file created with {len(submission)} rows")