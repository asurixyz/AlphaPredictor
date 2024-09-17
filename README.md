# Predicting Alpha Contest
This repository contains my submission for the Inter IIT Quant Guild Selection Contest [here](https://www.kaggle.com/competitions/taramani-quant-research-contest-tqrc/) on Kaggle.
This contest is part of the Inter-IIT Quant Guild selection process. It is the second of two problems presented for the selection, focusing on quantitative trading strategies using limit order book data.

## Files

- `code.ipynb`: Jupyter notebook containing the exploratory data analysis and model development process with detailed explanations of the features used.
- `code.py`: The complete Python script with all the code used in this project.

## Problem Statement

The challenge is to predict future returns in the next 1 second using limit order book (LOB) data. The goal is to build alphas (predictive signals) and use simple linear models to forecast short-term price movements.

## Strategy

Our approach involves:

1. Extensive feature engineering from LOB data
2. Ridge Regression with hyperparameter tuning
3. Feature scaling and missing value imputation

### Feature Engineering

We developed numerous features to capture market microstructure dynamics: (my apologies for the bad LaTeX)

1. **Price and Volume Imbalances**: 
   For each LOB level $i$, we calculate:
   
   $$\huge\text{Imbalance}_i = \frac{\text{BidVolume}_i - \text{AskVolume}_i}{\text{BidVolume}_i + \text{AskVolume}_i}$$

2. **Order Flow Imbalance**:
   
   $$\huge\text{OFI} = \frac{\text{RecentBuyOrders} - \text{RecentSellOrders}}{\text{TotalOrders}}$$

3. **Market Pressure**:
   
  $$\huge\text{Pressure} = \frac{\sum_{i=1}^5 (\text{BidVolume}i) - \sum{i=1}^5 (\text{AskVolume}i)}{\sum{i=1}^5 (\text{BidVolume}i) + \sum{i=1}^5 (\text{AskVolume}_i)}$$

4. **Volume-Weighted Average Price (VWAP)**:
   
   $$\huge\text{VWAP}_i = \frac{\text{BidPrice}_i \cdot \text{BidVolume}_i + \text{AskPrice}_i \cdot \text{AskVolume}_i}{\text{BidVolume}_i + \text{AskVolume}_i}$$

5. **Bid-Ask Slope**:
   
   $$\huge\text{Slope} = \frac{\text{AskPrice}_5 - \text{AskPrice}_1}{\text{BidPrice}_1 - \text{BidPrice}_5}$$

6. **Depth Imbalance**:
   
  $$\huge\text{DepthImbalance} = \frac{\sum_{i=1}^5 (\text{BidVolume}_i)}{\sum{i=1}^5 (\text{AskVolume}_i)}$$

### Model: Ridge Regression

We use Ridge Regression, which solves the following optimization problem:

$$\huge\min_{\beta} \left( \|y - X\beta\|_2^2 + \alpha \|\beta\|_2^2 \right)$$

where $y$ is the vector of future returns, $X$ is the feature matrix, $\beta$ are the coefficients, and $\alpha$ is the regularization parameter.

Hyperparameter tuning is performed using GridSearchCV to find the optimal $\alpha$.

### Feature Scaling and Missing Values

- StandardScaler is applied to normalize features: 

$$\huge z = \frac{x - \mu}{\sigma}$$

- Missing values are handled through forward and backward filling

## Evaluation Metric

The primary evaluation metric is the Pearson correlation coefficient between predicted and actual returns:

$$\huge r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

where $x_i$ are the predicted returns, $y_i$ are the actual returns, and $\bar{x}$ and $\bar{y}$ are their respective means.

## Resources

The competition provided:
- Training and test LOB datasets
- Detailed problem overview and evaluation metrics
- Sample submission code

Some of the features used in this project were inspired by the following research paper - Yin, J., & Wong, H. Y. (2023). The relevance of features to limit order book learning, available [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4226309). This paper provides valuable insights into feature engineering for limit order book data and has influenced our approach to this competition.

For more details, refer to the [competition page](https://www.kaggle.com/competitions/taramani-quant-research-contest-tqrc/).

## Requirements

Required Python libraries:
- numpy
- pandas
- scikit-learn
- scipy

Install via pip: `pip install numpy pandas scikit-learn scipy`

## Usage

1. Install required libraries
2. Download datasets from Kaggle
3. Run `code.py` or `code.ipynb` for analysis and model training

Note : Make sure you change the paths to the training and test data to the paths in your environment.

## License

This project is open-source under the MIT License.
