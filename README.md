# India Used Car Advisor

Predicting used car prices in the Indian market using data analysis and machine learning.

## Status
- [x] Load and clean the data
- [x] Exploratory analysis and charts
- [x] Feature engineering (log price, one-hot encoding)
- [x] Train and compare models
- [x] Error analysis
- [ ] Save the model
- [ ] AI layer (plain-language input)
- [ ] Streamlit app

## Data
CarDekho used car dataset from Kaggle (manishkr1754/cardekho-used-car-data), about 15,400 listings.
Cleaning: removed duplicates, rows with 0 seats, km_driven above 500,000, and rows where km_driven equals selling_price (likely data-entry errors). 15,213 rows remain.

## Approach
- Target: log of the selling price (prices are heavily right-skewed).
- Features: vehicle age, km driven, mileage, engine, max power, seats, plus one-hot encoded brand, fuel type, seller type and transmission.
- 80/20 train/test split.

## Results (held-out test set)

| Model | R2 (log price) | RMSE (log) | MAE (Rs) |
|---|---|---|---|
| Random Forest | 0.932 | 0.177 | 107,177 |
| Gradient Boosting | 0.921 | 0.191 | 126,935 |
| Linear Regression | 0.893 | 0.222 | 146,626 |

For the Random Forest, the median absolute error is about 10%, and 79% of predictions are within 20% of the actual price.

## Limitations
- A few listings have inconsistent age, mileage or specs, which hurts predictions. Removing 12 rows with implausible km per year changed R2 by about 0.002, so they were kept.
- Very old, cheap cars are under-represented, and the model tends to overestimate their prices.
- The data is a single snapshot from one source and does not capture city, condition or market trends.
- Gradient Boosting was not tuned.
