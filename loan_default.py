import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

'''
A project that determines (classsification problem)
whether a given person will default his loan payment.
'''

loans = pd.read_csv('loan_data.csv')

loans.info()
loans.describe()
loans.head()

def data_plot(loans):
	plt.figure(figsize=(10,6))
	loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
	                                              bins=30,label='Credit.Policy=1')
	loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
	                                              bins=30,label='Credit.Policy=0')
	plt.legend()
	plt.xlabel('FICO')
	plt.show()

	plt.figure(figsize=(10,6))
	loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
	                                              bins=30,label='not.fully.paid=1')
	loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
	                                              bins=30,label='not.fully.paid=0')
	plt.legend()
	plt.xlabel('FICO')
	plt.show()

	plt.figure(figsize=(11,7))
	sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')

	sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')

	plt.figure(figsize=(11,7))
	sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
	           col='not.fully.paid',palette='Set1')
	plt.show()

data_plot(loans)

#Setting up the Data
loans.info()

cat_feats = ['purpose']

final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

final_data.info()

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))