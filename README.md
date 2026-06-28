# 🚲 Bike Sharing Demand Prediction

A Machine Learning project that predicts hourly bike rental demand using regression models and feature engineering on the Kaggle Bike Sharing Demand dataset.

---

## 📌 Overview

This project compares multiple regression models to analyze how increasing model complexity affects prediction performance.

Implemented models include:

* Linear Regression
* Polynomial Regression (Degree 2)
* Polynomial Regression (Degree 3)
* Polynomial Regression (Degree 4)
* Quadratic Regression with Interaction Features

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📊 Feature Engineering

* Converted `datetime` into **Hour** and **Month**
* Removed data leakage features (`casual`, `registered`)
* Standardized numerical features using **StandardScaler**

---

## 📈 Results

| Model                                       |   R² Score |
| ------------------------------------------- | ---------: |
| Linear Regression                           |     0.3438 |
| Polynomial Regression (Degree 2)            |     0.4455 |
| Polynomial Regression (Degree 3)            |     0.4910 |
| Polynomial Regression (Degree 4)            |     0.4934 |
| Quadratic Regression (Interaction Features) | **0.4984** |

### Key Achievements

* Improved **R² score by approximately 45%** compared to the baseline Linear Regression model.
* Reduced **Mean Squared Error (MSE) by approximately 24%**.
* Demonstrated the effectiveness of feature engineering and interaction terms in improving predictive performance.

---

## 📂 Repository Structure

```text
Bike-Sharing-Demand-Prediction/
│
├── bike_demand_prediction.py
├── train.csv
├── Project_Report.pdf
└── README.md
```

---

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib scikit-learn

python bike_demand_prediction.py
```

---

## 📄 Report

The complete project report explaining the methodology, preprocessing, model comparison, and results is available in **Project_Report.pdf**.

---

## 👨‍💻 Author

**Rishi Raj Nalla**
