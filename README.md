# Capstone Submission: Predicting Ski Season Length for Lake Tahoe Resorts

![Skier Daffy Image](Images/Skier_Daffy.png)

## Executive Summary

This project explores how machine learning models can predict optimal snow conditions and season length for Lake Tahoe ski resorts by analyzing historical weather patterns and climate indicators. Using data from multiple sources, we identified the key factors influencing ski season duration in the Lake Tahoe region.

We compared three different regression models (Random Forest, Gradient Boosting, and Support Vector Regression) and found that the Random Forest model achieved the best predictive capability with an R² score of 0.36, explaining a significant portion of the variance in season length despite the inherent complexity and variability of weather systems.

The most important predictors of season length were:
1. Pre-season freeze days (importance 0.55)
2. Resort-specific characteristics, particularly for Palisades (importance 0.31)
3. Pre-season snow accumulation (importance 0.06)
4. Pre-season precipitation (importance 0.03)
5. Seasonal ONI (El Niño/La Niña) values (importance 0.03)

These findings provide actionable insights for ski resort operations, marketing strategies, and long-term planning in the context of changing climate patterns.

## Research Question

How can machine learning models predict optimal snow conditions and season length for Lake Tahoe ski resorts by analyzing historical weather patterns and climate indicators?

## Background and Motivation

Ski resorts operate in an environment of significant uncertainty due to weather dependence. Accurate predictions of season length can help resorts:

- Optimize staffing and operational planning
- Develop more effective marketing and pricing strategies
- Make informed infrastructure investment decisions (especially for snowmaking)
- Adapt to changing climate patterns

As climate change affects traditional weather patterns, the ability to predict season characteristics becomes increasingly valuable for the winter sports industry.

## Data Sources

The analysis combines three primary datasets:

### NOAA Weather Station Data
- Daily observations from multiple stations around Lake Tahoe (2014-2025)
- Includes temperature (max, min, avg), precipitation, snowfall, and snow depth
- Stations at various elevations around the Tahoe Basin

### Ski Resort Season Records
- Historical opening and closing dates for four major Tahoe resorts:
  - Heavenly
  - Palisades
  - Sugar Bowl
  - Kirkwood
- Records for seasons from 2014-2015 through 2024-2025

### NOAA Oceanic Niño Index (ONI)
- Measures El Niño/La Niña climate patterns
- Three-month running mean of sea surface temperature anomalies
- Important climate indicator that influences Western US winter weather

## Methodology

### Data Processing and Feature Engineering

1. **Date transformations**: Created consistent season identifiers and winter day calculations
2. **Rolling window statistics**: Calculated 7-day, 14-day, and 30-day rolling averages for key weather metrics
3. **Seasonal cumulative metrics**: Tracked total snowfall, precipitation, and freezing days
4. **Pre-season conditions**: Analyzed weather patterns from November 1 to resort opening
5. **Climate integration**: Merged ONI data with local weather observations

### Exploratory Data Analysis

Key relationships explored:
- Impact of El Niño/La Niña conditions on season length
- Correlation between pre-season snow and opening dates
- Relationship between total seasonal snowfall and season duration
- Comparison of season patterns across different resorts

### Modeling Approach

We implemented and compared three different regression models:

1. **Random Forest Regression**
   - Ensemble of decision trees that captures non-linear relationships
   - Naturally models complex interactions between features
   - Provides built-in feature importance measures
   - Resistant to overfitting with limited data

2. **Gradient Boosting Regression**
   - Sequentially builds trees to correct errors from previous ones
   - Often achieves high accuracy for regression problems
   - Captures subtle patterns in the data
   - Provides interpretable feature importance

3. **Support Vector Regression (SVR)**
   - Uses a kernel approach to map data into higher-dimensional space
   - Offers a fundamentally different approach than tree-based methods
   - Provides a useful benchmark for comparison

For each model, we:
- Selected features based on domain knowledge and exploratory analysis
- One-hot encoded resort identifiers to capture resort-specific effects
- Used standard scaling to normalize feature ranges
- Performed hyperparameter tuning via grid search
- Evaluated performance using RMSE, MAE, and R² metrics

## Results

### Model Performance Comparison

Our model evaluation showed that Random Forest outperformed the other approaches:

| Model              | RMSE (days) | MAE (days) | R² Score |
|--------------------|-------------|------------|----------|
| Random Forest      | 26.96       | 20.97      | 0.36     |
| Gradient Boosting  | 31.48       | 24.05      | 0.13     |
| SVR                | 33.45       | 26.68      | 0.02     |

The Random Forest model demonstrated significantly better predictive power, explaining 36% of the variance in season length, while Gradient Boosting explained 13% and SVR only 2%.

### Key Findings

1. **Pre-season freeze days** show the strongest correlation with season length (importance 0.55), highlighting the importance of consistent cold temperatures for establishing a base and enabling snowmaking.

2. **Resort-specific factors** significantly impact season length, with Palisades showing distinct patterns (importance 0.31), likely due to elevation, aspect, or operational strategies.

3. **Climate oscillations** (El Niño/La Niña) demonstrate measurable influence on Tahoe ski seasons (importance 0.03). Generally, La Niña conditions (negative ONI) are associated with longer ski seasons, while El Niño conditions (positive ONI) tend to result in shorter seasons.

4. **Pre-season snowfall** impacts resort opening dates (importance 0.06), with more early-season snow generally leading to earlier openings and potentially longer seasons.

5. **Pre-season precipitation** (importance 0.03) represents potential snowfall when temperatures are cold enough, contributing to early-season conditions.

### Feature Importance Agreement

All three models consistently identified pre-season freeze days and resort-specific characteristics (particularly Palisades) as the most important predictors, providing strong confidence in these findings despite differences in modeling approaches.

## Implications

The findings suggest that:

1. **Pre-season conditions matter most**: The weather patterns before resorts open have the strongest influence on overall season length, suggesting that early-season investments in snowmaking during cold periods could significantly extend seasons.

2. **Climate indicators provide early signals**: ONI values can offer insights months before the ski season, potentially allowing for long-range planning and marketing strategies.

3. **Resort-specific characteristics create resilience**: Location, elevation, and operational factors create distinct advantages for certain resorts, which could inform long-term investment decisions.

4. **Opening and closing dates have different drivers**: Resorts should consider different sets of factors when planning for opening versus extending their seasons into spring.

## Limitations

- Limited historical data (only covering seasons since 2014-2015)
- Weather station observations may not perfectly represent conditions at specific resort locations
- Simplified representation of complex weather systems
- Some aspects of season length are operational decisions rather than purely condition-dependent

## Next Steps

Potential extensions of this analysis include:

1. **Incorporating additional climate indicators**: Pacific Decadal Oscillation, atmospheric rivers
2. **Developing separate models** for opening and closing date predictions
3. **Adding snowpack quality metrics** beyond simple depth measurements
4. **Expanding to include more resorts** across different geographic regions
5. **Exploring additional model architectures**: Neural networks, ensemble approaches combining multiple model types
6. **Exploring time-series approaches** (ARIMA, LSTM) for more granular predictions

## Project Structure

This project is organized into two Jupyter notebooks:

1. [**Capstone_Notebook1_DataProcessing_and_EDA.ipynb**](https://github.com/cjplumer/module24_capstone_finalreport/blob/main/Capstone_Notebook1_DataProcessing_and_EDA.ipynb) - Data processing, feature engineering, and exploratory data analysis
2. [**Capstone_Notebook2_Modeling_and_Evaluation.ipynb**](https://github.com/cjplumer/module24_capstone_finalreport/blob/main/Capstone_Notebook2_Modeling_and_Evaluation.ipynb) - Model development, comparison, evaluation, and interpretation

Intermediate processed datasets are saved in the `processed_data` directory, and the trained models are saved in the `models` directory.

## Installation and Usage

### Prerequisites
- Python 3.8+
- Required Python packages:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - shap

### Setup
1. Clone this repository
2. Install required packages
3. Run the notebooks in order:
 - First run `Capstone_Notebook1_DataProcessing_and_EDA.ipynb`
 - Then run `Capstone_Notebook2_Modeling_and_Evaluation.ipynb`

### Data
Place the required datasets in the project root directory:
- `Tahoe_NOAA_Data.csv`
- `Tahoe_Resort_Season_Dates.csv`
- `NOAA_ONI_Data.csv`
