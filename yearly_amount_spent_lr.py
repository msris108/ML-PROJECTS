import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv("Ecommerce Customers")

def data_plot(customers):
	customers.head()
	customers.describe()
	customers.info()

	sns.set_palette("GnBu_d")
	sns.set_style('whitegrid')

	sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
	sns.show()

	sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
	sns.show()

	sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
	sns.show()

	sns.pairplot(customers)
	sns.show()

	sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
	sns.show()

#data_plot(customers)

#Training and Testing Data

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



lm = LinearRegression()
lm.fit(X_train,y_train)
#print('Coefficients: \n', lm.coef_)
predictions = lm.predict( X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Residuals
sns.distplot((y_test-predictions),bins=50);