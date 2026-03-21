## 🍉🍈🍇 Linear Regression 🍇🍈🍉
- The model calculates a specific "slope" and an "intercept" to define the best-fit line. The slope represents the average increase in house price for every additional room added to the property
- Predict the price of a house based on the number of rooms.
- Scikit-Learn requires the independent variable (X) to be a 2-dimensional array. Even with a single feature like "bedrooms," the data must be reshaped so the model recognizes it as a matrix of features rather than a flat list.
- Multicollinearity Issues: When using multiple predictors, you must ensure they aren't too highly correlated with each other. If "number of rooms" and "total square feet" move perfectly in sync, the model struggles to determine which variable is actually driving the price change.
- Extrapolation Risks: Linear regression is great at predicting within the range of your data, but it can be dangerous when predicting outside of it. For instance, the model might unrealistically predict a 100-room house costs a billion dollars, even if such a property doesn't exist in reality.
- The Mean Absolute Error (MAE): While R-squared tells you the "fit" quality, MAE provides a more relatable metric by showing the average dollar amount your predictions are off by. It’s often easier to explain to stakeholders that the model is "off by $15,000 on average" than quoting a decimal score.
- Feature Scaling: While simple linear regression isn't strictly required to have scaled data, it becomes vital once you move to Multiple Linear Regression with different units. Scaling ensures that a variable like "square footage" (thousands) doesn't numerically overwhelm "number of bathrooms" (single digits).
- Assumption of Normality: The model performs best when the "residuals" (the differences between actual and predicted values) follow a normal distribution. If your errors are skewed, it suggests the model might be missing a pattern or that the data needs a transformation.
- Before choosing this model, it is vital to confirm that a linear relationship actually exists between your variables. If the data follows a curve rather than a straight line, a simple linear model will "underfit" and provide poor predictions.
- The Identity of Residuals: A residual is the vertical distance between a data point and the regression line, representing the "error" for that specific observation. Analyzing a residual plot is a pro move to check if your linear assumption holds or if there’s hidden heteroscedasticity.
- The same principles used here can be expanded to include more variables, such as square footage or the age of the house. This is known as Multiple Linear Regression, which allows for a much more comprehensive and accurate pricing model.
- Linear regression can be highly sensitive to "outliers," which are data points that fall far from the general trend. A single luxury mansion with a very low room count could significantly tilt your regression line and reduce overall accuracy.
- To know how well your model actually performs, you should check the R-squared score. This value indicates the proportion of the variance for the house price that's explained by the number of bedrooms.
- Under the hood, this model typically uses Ordinary Least Squares (OLS) to find the best line. It works by minimizing the sum of the squares of the vertical deviations between each actual data point and the predicted line.

```
import numpy as np
```
 
## 🍉🍈🍇 Create Data on The Number of Rooms 🍇🍈🍉
First we import the required libraries. Then create dummy data using a numpy array.

```
bedrooms = np.array([1,1,2,2,3,4,4,5,5,5])
```
 
## 🍉🍈🍇 House Price Data 🍇🍈🍉
Assumptions in dollars.Next, we can try to display the data in the form of a scatter plot. The number of rooms on the X axis is the independent variable and the house price on the Y axis is the dependent variable. 

```
house_price = np.array([15000, 18000, 27000, 34000, 50000, 68000, 65000, 81000,85000, 90000])
```

The display of the code is as follows.
![image](https://github.com/diantyapitaloka/Sklearn-Linearregression/assets/147487436/e238c50e-0f5e-4f3e-9ce4-0bd3c763dcb4)

## 🍉🍈🍇 Train The Model with Linear Regression Fit 🍇🍈🍉
Then in the next cell, we can start training our model by calling the LinearRegression.fit() function on our data. This function is for training a linear regression model from the SKLearn library.

```
from sklearn.linear_model import LinearRegression
bedrooms = bedrooms.reshape(-1, 1)
linreg = LinearRegression()
linreg.fit(bedrooms, house_price)
```

## 🍉🍈🍇 Displays a Plot of The Relationship Between the Number of Rooms and The House Price 🍇🍈🍉
Finally we can see how our model fits the data we have by making a plot of our model.
```
plt.scatter(bedrooms, house_price)
plt.plot(bedrooms, linreg.predict(bedrooms))
```

## 🍉🍈🍇 Output 🍇🍈🍉
Linear regression models are one of the simplest machine learning models. This model has low complexity and works very well on datasets that have linear relationships. So, when you encounter a problem that appears to have a linear relationship, linear regression can be the first choice as a model to develop.

![image](https://github.com/diantyapitaloka/Sklearn-Linearregression/assets/147487436/5822b1a3-2220-4072-89f4-c723caf4b563)

## 🍉🍈🍇 License 🍇🍈🍉
- Copyright by Diantya Pitaloka
