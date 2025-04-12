# Car Price Prediction Using Regression Models

## Overview

This project uses multiple regression models to predict car prices based on various features such as engine size, horsepower, fuel type, and others. The models used include:

-Linear Regression
-Ridge Regression
-Lasso Regression
-Random Forest Regressor
-Gradient Boosting Regressor

The data undergoes preprocessing (including feature scaling and one-hot encoding), and the models are evaluated based on performance metrics like RMSE (Root Mean Squared Error) and R² score.

## Requirements

To run this project, you’ll need to install the following Python libraries:

-pandas
-numpy
-matplotlib
-seaborn
-scikit-learn

You can install the required libraries using `pip`:

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

## Project Structure

-**data/**: Folder containing the dataset file (`car_price_dataset.csv`)
-**main.py**: Python script where data is loaded, preprocessed, models are trained, and results are evaluated.
-**requirements.txt**: List of dependencies for the project.
-**README.md**: This file.

## Dataset

The dataset consists of information about various car models. The features include:

-Fuel Type
-Engine Size
-Horsepower
-Car Body Type
-Wheelbase
-Car Dimensions (Length, Width, Height)

The target variable is the car price.

## Steps and Description

1. Loading and Exploring the Data

    ```bash
    data = pd.read_csv("car_price_dataset.csv")
    print(data.info())
    print(data.isnull().sum())
    ```

The dataset is loaded, and basic information like missing values and data types is displayed.

2. Preprocessing the Data

-Missing values are dropped.
-Categorical variables are one-hot encoded.
-Numerical variables are standardized using StandardScaler.

3. Modeling

Five different regression models are used:

-Linear Regression
-Ridge Regression
-Lasso Regression
-Random Forest Regressor
-Gradient Boosting Regressor

These models are evaluated using RMSE and R² metrics.

4. Train-Test Split

    ```bash
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

The data is split into training and testing sets (80% training, 20% testing).

5. Evaluation

The models are evaluated based on their Root Mean Squared Error (RMSE) and R² score to determine their prediction accuracy.

6. Feature Importance Visualization

Using Random Forest Regressor, feature importances are visualized to understand which features are most significant in predicting car prices.

    ```bash
    importances = pipeline.named_steps['model'].feature_importances_
    ```

This will plot a bar chart of the most important features.

7. Model Performance Comparison

After training the models, we compare their performance:

    ```bash
    results_df = pd.DataFrame(results).T
    print(results_df.sort_values(by='RMSE'))
    ```

8. Results

The model performances (RMSE and R² scores) are printed and visualized in sorted order, showing which model performs best.

## How to Run the Project

1. Clone the Repository

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install Dependencies

    ```bash
    pip install -r requirements.txt
    ```

4, Run the Python Script

    ```bash
    python main.py
    ```

Results will be printed in the terminal, including model evaluation metrics and feature importance visualization.

## Visualization of Feature Importance

Here’s the Python code for visualizing the feature importances of the trained models (specifically Random Forest):

    ```bash
    importances = pipeline.named_steps['model'].feature_importances_
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    ohe_features = ohe.get_feature_names_out(categorical_features)
    all_features = numerical_features + list(ohe_features)
  
    # Create a DataFrame
    feature_importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    feature_importance_df.sort_values(by='Importance', ascending=True, inplace=True)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Visualizing Important Features')
    plt.tight_layout()
    plt.show()
    ```

## Evaluation Metrics

The evaluation of the models is done using the following metrics:

-**Root Mean Squared Error (RMSE)**: Measures the average magnitude of the prediction errors.
-**R² Score**: Indicates how well the regression model predicts the target variable (car price).

## Conclusion

In this project, I applied different regression models to predict car prices. After preprocessing the data and training the models, we evaluated their performances. Based on RMSE and R² scores, we identified the best-performing model.
