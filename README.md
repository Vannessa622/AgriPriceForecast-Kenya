![The Header](Data/Group7.jpg)

# AgriPriceForecast Kenya
AgriPriceForecast Kenya is a comprehensive data science project that leverages advanced forecasting techniques and sentiment analysis to provide actionable insights into Kenya’s agricultural market. The project aims to empower farmers, traders, policymakers, and other stakeholders by predicting commodity price trends and analyzing market sentiment through social media data.

* Table of Contents
* Overview
* Business Understanding
* Project Objectives
* Data Sources
* Data Cleaning and Processing
* Modeling and Forecasting
* Sentiment Analysis
* Usage
* Tableau Dashboard

AgriPriceForecast Kenya is designed to analyze historical agricultural market data and forecast future commodity prices using advanced machine learning models. The project also incorporates sentiment analysis from social media to capture public perceptions that influence market behavior. Through these combined approaches, the system delivers insights that help in market planning, price stabilization, and informed decision-making for all stakeholders in Kenya's agricultural sector.

## Business Understanding
The agricultural sector in Kenya is a critical component of the economy, with significant contributions to livelihoods. However, the market is subject to:

* Price Volatility: Seasonal fluctuations and supply chain inefficiencies lead to unpredictable commodity prices.
Market Inefficiencies: Differences in regional market conditions and infrastructure challenges affect pricing.
Public Sentiment: Consumer and stakeholder perceptions, often expressed on social media, can influence market dynamics.

By addressing these challenges, AgriPriceForecast Kenya aims to:

* Enable farmers and traders to time their sales for maximum profit.
* Inform government policies for market stabilization.
* Provide financial institutions with data-driven insights for risk assessment.

## Project Objectives
* Assess Commodity Price Fluctuations:

* Analyze historical data to evaluate price volatility across various commodities and regions.
Identify factors such as seasonal variations, supply shortages, and logistical challenges.
Analyze Market Trends:

* Monitor and visualize market trends over time.
Develop predictive models to forecast price movements using both short-term and long-term strategies.
Incorporate Sentiment Analysis:

* Use social media data (e.g., tweets) to gauge public sentiment regarding commodity prices.
Integrate sentiment insights with price forecasts to enhance prediction accuracy.

## Data Sources
*Kenya Agricultural Market Information System (KAMIS):*
Historical commodity prices, trade volumes, and market highlights for various products such as maize, onions, cabbages, etc.

*Geographical Data:*
Market locations, county information, and regional characteristics.

*Social Media Data:*
Tweets and other social media posts analyzed using VADER for sentiment scores.

## Data Cleaning and Processing
The project includes comprehensive data cleaning steps:

*Data Integration:*
Aggregation of multiple CSV files containing commodity prices.
*Data Transformation:*
Conversion of price fields from string format to float, extraction of units, and calculation of markup percentages.
*Outlier Treatment:*
Identification and removal of unrealistic price and supply volume values.
*Handling Missing Values:*
Imputation of missing supply volumes using commodity-specific medians and removal of records with missing geographical coordinates.
*Data Normalization:*
Standardizing date formats and categorizing commodities for simplified analysis.
## Modeling and Forecasting
Two advanced models were evaluated for forecasting:

*XGBoost:*
Provided competitive performance for short-term predictions.
Captured abrupt price changes well but was less effective in modeling long-term trends.

*LSTM (Long Short-Term Memory):*
Outperformed XGBoost in modeling sequential dependencies and long-term trends.
Was selected as the final forecasting model due to its superior ability to capture seasonal patterns and recurring price cycles.

## Sentiment Analysis
*VADER Sentiment Analysis:*
Applied to social media text to compute sentiment scores ranging from -1 (most negative) to +1 (most positive).
Analysis revealed that while public sentiment is generally mildly positive (average score ~0.17), fluctuations in sentiment are closely linked to market events and price changes.

## Usage

*Data Processing & Analysis:*
Run the main notebook Project.ipynb to execute data cleaning, exploratory data analysis, model training, and sentiment analysis.

*Forecasting:*
Use the forecasting module to generate price predictions with the LSTM model.
To start the API, run the app.py file in terminal with the code commented out at the bottom of the file.

*Visualization:*
Explore generated plots and dashboards for detailed market insights.

*Dependencies*
install cleanrequirements.txt to install all the libraries necessary for this project

## Tableau Dashboard
https://public.tableau.com/views/AGRI-FORECASTKENYA/AGRI-FORECASTKENYA_?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link