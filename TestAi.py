import numpy as np
from sklearn.datasets import load_iris
from sklearn import linear_model,datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plott
X,y = load_iris(return_X_y=True)

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

   # Use only one feature
   diabetes_X = diabetes_X[:, np.newaxis, 2]

   # Split the data into training/testing sets
   diabetes_X_train = diabetes_X[:-20]
   diabetes_X_test = diabetes_X[-20:]

   # Split the targets into training/testing sets
   diabetes_y_train = diabetes_y[:-20]
   diabetes_y_test = diabetes_y[-20:]

   # Create linear regression object
   #regr = linear_model.LinearRegression()
   regr = MLPRegressor(random_state=1, max_iter=500).fit(diabetes_X_train, diabetes_y_train)
   # Train the model using the training sets
   #regr.fit(diabetes_X_train, diabetes_y_train)

   # Make predictions using the testing set
   diabetes_y_pred = regr.predict(diabetes_X_test)

   # The coefficients
   print('Coefficients: \n', regr.coef_)
   # The mean squared error
   print('Mean squared error: %.2f'
         % mean_squared_error(diabetes_y_test, diabetes_y_pred))
   # The coefficient of determination: 1 is perfect prediction
   print('Coefficient of determination: %.2f'
         % r2_score(diabetes_y_test, diabetes_y_pred))

   # Plot outputs
   plott.scatter(diabetes_X_test, diabetes_y_test, color='black')
   plott.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

   plott.xticks(())
   plott.yticks(())

   plott.show()