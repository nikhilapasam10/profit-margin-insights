# profit-margin-insights

## 1.1 Information About the Project
This project analyzes and models an e-commerce sales transactions dataset containing 34,500 records and 17 features. The dataset simulates online shopping behavior and can be used for several machine learning related tasks.

**Objective:**  
The main objective of this project is to build a machine learning model to predict profit margin. Since the target variable (profit margin) is continuous, this is a regression task. Predicting profit can help businesses to improve pricing strategies, optimize discounts, and reduce overall losses.

**Scope:**  
This project includes exploratory data analysis, feature engineering, data preprocessing, model training and evaluation, and building a deployable machine learning model.

## 1.2 Description of the Dataset
The dataset, titled E-Commerce Sales Transactions Dataset, was obtained from Kaggle. 

- **Size:** 34,500 rows (records) and 17 columns (features)
- **Type:** Tabular dataset that contains numerical features, categorical features, and date-time data

## 1.3 Description of the Columns

- **Target Variable:** 
    - profit_margin (continuous numerical variable that represents the profit earned from each order)
- **Feature Variables:** 
    - order_id (categorial variable that represents a unique identifier for each order)
    - customer_id (categorical variable that represents a unique identifier for each customer)
    - product_id (categorical variable that represents a unique identifier for each product)
    - category (nominal categorical variable that represents the product category)
    - price (continuous numerical variable that represents the unit price of the product)
    - discount (continuous numerical variable that represents the discount applied (%))
    - quantity (discrete numerical variable that represents the number of items purchased)
    - customer_age (continuous numerical variable that represents the age of the customer)
    - customer_gender (nominal categorical variable that represents the gender of the customer)
    - payment_method (nominal categorical variable that represents the payment type)
    - region (nominal categorical variable that represents the geographic region of the customer)
    - order_date (datetime variable that represents the date when the order was placed)
    - delivery_time_days (numerical variable that represents the number of days taken for delivery)
    - total_amount (continuous numerical variable that represents the final amount paid after discounts)
    - shipping_cost (continuous numerical variable that represents the delivery charges)
    - returned (categorical variable that represents whether the product was returned or not)
