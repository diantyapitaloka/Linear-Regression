# Sklearn-Linearregression
Predict the price of a house based on the number of rooms.

- import numpy as np
 
## Create Data on The Number of Rooms
First we import the required libraries. Then create dummy data using a numpy array.

- bedrooms = np.array([1,1,2,2,3,4,4,5,5,5])
 
## House Price Data
Assumptions in dollars.Next, we can try to display the data in the form of a scatter plot. The number of rooms on the X axis is the independent variable and the house price on the Y axis is the dependent variable. 

- house_price = np.array([15000, 18000, 27000, 34000, 50000, 68000, 65000, 81000,85000, 90000])

The display of the code is as follows.
![image](https://github.com/diantyapitaloka/Sklearn-Linearregression/assets/147487436/e238c50e-0f5e-4f3e-9ce4-0bd3c763dcb4)

## Train The Model with Linear Regression Fit
Then in the next cell, we can start training our model by calling the LinearRegression.fit() function on our data. This function is for training a linear regression model from the SKLearn library.
- from sklearn.linear_model import LinearRegression

- bedrooms = bedrooms.reshape(-1, 1)
- linreg = LinearRegression()
- linreg.fit(bedrooms, house_price)

## Displays a Plot of The Relationship Between the Number of Rooms and The House Price
Finally we can see how our model fits the data we have by making a plot of our model.
- plt.scatter(bedrooms, house_price)
- plt.plot(bedrooms, linreg.predict(bedrooms))

Linear regression models are one of the simplest machine learning models. This model has low complexity and works very well on datasets that have linear relationships. So, when you encounter a problem that appears to have a linear relationship, linear regression can be the first choice as a model to develop.

![image](https://github.com/diantyapitaloka/Sklearn-Linearregression/assets/147487436/5822b1a3-2220-4072-89f4-c723caf4b563)
