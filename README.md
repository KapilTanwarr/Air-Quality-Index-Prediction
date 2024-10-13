# Air-Quality-Index-Prediction

This project focuses on predicting the Air Quality Index (AQI) using various machine learning models. 
The dataset consists of air quality parameters such as PM2.5, PM10, NO2, SO2, O3, CO, Benzene, Toluene, and other pollutants. 
The goal is to analyze the data and predict AQI to understand how polluted the air is.

Project Overview
Air pollution is a significant issue, and the Air Quality Index (AQI) helps quantify the level of pollution and its potential impact on human health. 
This project uses machine learning techniques to predict AQI values based on different air pollutants.

Dataset
The dataset used in this project contains historical air quality data with various features such as:

PM2.5: Fine particulate matter
PM10: Particulate matter with a diameter of 10 micrometers or less
NO2: Nitrogen dioxide levels
SO2: Sulfur dioxide levels
O3: Ozone concentration
CO: Carbon monoxide levels
Benzene, Toluene, Xylene: Volatile organic compounds

Requirements
Make sure you have the following installed:

Python 3.x
Jupyter Notebook
pandas
numpy
matplotlib
seaborn
scikit-learn

Project Workflow

Step 1: Data Loading and Preprocessing
Loading the dataset: We load the dataset containing air quality parameters.
Handling missing values: We check for any missing data and handle it appropriately, either through imputation or removal of rows with null values.
Feature engineering: Additional features are created from the existing columns, such as date-time features or log-transformation of pollutants for better model performance.

Step 2: Exploratory Data Analysis (EDA)
Visualizing air pollutants: Using libraries like matplotlib and seaborn, we plot graphs showing trends and distributions of the pollutants over time.
Correlation analysis: We create a heatmap to understand the correlation between various pollutants and AQI to see which factors impact air quality the most.

Step 3: Model Selection
We use several machine learning models to predict AQI:

Linear Regression
Decision Tree
Random Forest
Gradient Boosting
Cross-validation and hyperparameter tuning are applied to optimize the models and achieve the best prediction accuracy.

Step 4: Model Training and Evaluation
Model training: The dataset is split into training and testing sets. Models are trained using the training set.
Evaluation metrics: We use metrics such as Mean Squared Error (MSE) and R-squared to evaluate the performance of each model.
Best model: After testing multiple models, we select the model with the best performance for predicting AQI.

Step 5: Results and Visualization
Predicted vs Actual AQI: We plot graphs showing the predicted AQI values against the actual AQI values to visualize the model’s performance.
Feature importance: For models like Random Forest, we display the most important features influencing AQI predictions.
