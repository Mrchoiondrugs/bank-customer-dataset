# Bank Customer Churn Analysis & Prediction

This repository contains a comprehensive deep learning pipeline designed to predict customer churn for a retail banking institution. Using an Artificial Neural Network (ANN) built with TensorFlow/Keras, the model analyzes various customer demographic, financial, and behavioral attributes to identify individuals at high risk of leaving the bank.

---

## 1. Executive Summary
Customer retention is one of the primary drivers of profitability in the retail banking sector. Acquiring a new customer can cost significantly more than retaining an existing one. This project leverages an Artificial Neural Network (ANN) to analyze data from bank customers and accurately predict the likelihood of a customer churning (`churn = 1`). 

By preemptively flagging high-risk customers, the bank's marketing and customer relationship teams can deploy targeted loyalty programs, financial incentives, or proactive customer service interventions to mitigate customer attrition.

---

## 2. Dataset Overview & Data Dictionary

The analysis is based on a dataset containing structural characteristics of bank customers. The features span demographic data, financial account statuses, and behavioral engagement indicators.

### Feature Descriptions
| Column Name | Data Type | Type | Description |
| :--- | :--- | :--- | :--- |
| `customer_id` | Integer | Identifier | A unique identifier for each customer. Dropped during training. |
| `credit_score` | Integer | Quantitative | The credit score of the customer, reflecting creditworthiness. |
| `country` | String | Categorical | The customer's country of residence (e.g., France, Spain, Germany). |
| `gender` | String | Categorical | The customer's gender (Female or Male). |
| `age` | Integer | Quantitative | The customer's age in years. |
| `tenure` | Integer | Quantitative | The number of years the customer has been with the bank. |
| `balance` | Float | Quantitative | The current account balance held by the customer. |
| `products_number`| Integer | Quantitative | The number of distinct financial products the customer utilizes. |
| `credit_card` | Integer | Binary | Indicates whether the customer possesses a credit card (`1 = Yes`, `0 = No`). |
| `active_member` | Integer | Binary | Indicates active user status based on recent transactions (`1 = Active`). |
| `estimated_salary`| Float | Quantitative | The estimated annual salary of the customer. |
| `churn` | Integer | Binary | **Target Variable**: Indicates if the customer closed their account (`1 = Churned`, `0 = Retained`). |

---

## 3. Exploratory Data Analysis (EDA) & Domain Observations

Based on standard structural reviews of this dataset profile, several critical customer behaviors and data characteristics stand out:

* **The Demographic Influence of Age:** Customer age is typically one of the strongest statistical predictors of churn. Older customers exhibit a significantly higher churn rate compared to younger demographics. This suggests that the bank's product offerings or user experience may not be keeping pace with the changing financial needs of mid-to-late career clients.
* **The "Zero-Balance" Phenomenon:** A substantial portion of the customer base shows a `balance` of exactly `0.00` (as seen in rows 0 and 3 of the sample preview). This indicates a segment of dormant or secondary accounts. Interestingly, zero-balance customers often have different churn dynamics compared to active, high-balance clients, requiring segmented retention strategies.
* **Geographic Variations:** Churn rates vary notably across different countries. For instance, European banking datasets frequently reveal that customers in specific regions (such as Germany) churn at double the rate of adjacent markets like France or Spain, potentially pointing to aggressive regional competitor campaigns or localized service dissatisfaction.
* **Product Over-Saturation:** While utilizing multiple products generally improves customer stickiness, a sharp tipping point exists. Customers holding more than 2 products (e.g., `products_number >= 3`) display an exponentially higher propensity to churn. This counter-intuitive trend often signifies customer frustration from cross-selling irrelevant products or unresolved compound issues across accounts.
* **Engagement Mitigation:** Being an `active_member` acts as a strong buffer against churn. Inactive members are substantially more susceptible to closing their accounts, indicating that ongoing digital or transactional engagement is vital.

---

## 4. Feature Engineering & Preprocessing Pipeline

To prepare the raw tabular data for optimal deep learning performance, a rigorous preprocessing pipeline is executed:

1. **Identifier Elimination:** The `customer_id` column is dropped because it represents a random unique key with no mathematical or contextual predictive power. Leaving it would cause the neural network to overfit on noise.
2. **Categorical Encoding:** Neural networks require strictly numerical inputs. Non-numeric columns (`country`, `gender`) are converted into structural numeric features using **One-Hot Encoding**. The `drop_first=True` argument is utilized to omit the first dummy category, successfully mitigating the multi-collinearity trap (dummy variable trap).
3. **Train-Test Stratification:** The dataset is split into an 80% training set and a 20% testing set. A fixed `random_state` ensures that evaluation metrics remain perfectly reproducible across different script runs.
4. **Feature Scaling (Normalization):** Quantitative values possess drastically different scales (e.g., `age` ranges from 18–90, while `estimated_salary` and `balance` exceed 100,000). To prevent high-magnitude features from dominating the network's gradient updates, `StandardScaler` is applied to standardize features to a mean of 0 and a variance of 1.

---

## 5. Neural Network Architecture & Design

The predictive model is implemented as a sequential Artificial Neural Network (ANN) designed to extract hierarchical interactions from tabular inputs:

```python
model = Sequential([
    # Input Layer + Hidden Layer 1
    Dense(units=16, activation='relu', input_shape=(X_train.shape[1],)),
    
    # Hidden Layer 2
    Dense(units=8, activation='relu'),
    
    # Output Layer
    Dense(units=1, activation='sigmoid')
])
