# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters
2. Compute Sigmoid Predictions
3. Update Parameters Using Gradient Descent
4. Predict Output

## Program:
```
import numpy as np
import pandas as pd

# Dataset
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Feature matrix X and target y
X = df[['Hours_Studied', 'Previous_Score', 'Internship']].values
y = df['Placement'].values

m, n = X.shape

# Add bias term
X_b = np.c_[np.ones((m, 1)), X]

# Initialize theta (weights)
theta = np.zeros(n + 1)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta = theta - alpha * gradient
    return theta

# Train the model
alpha = 0.01
iterations = 5000
theta = gradient_descent(X_b, y, theta, alpha, iterations)

print("Final Theta Values:", theta)

/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:  
*/
```

## Output:

<img width="1036" height="55" alt="image" src="https://github.com/user-attachments/assets/87aa9ce3-4abf-463e-90e7-b4b89fbfcde3" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

