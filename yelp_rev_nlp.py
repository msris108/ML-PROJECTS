import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import  TfidfTransformer

'''
NLP project on YELP REVIEW CLASSIFICATION
1 star or 5 star 

>>piplining is used and verified

'''

yelp = pd.read_csv('yelp.csv')

yelp.head()
yelp.info()
yelp.describe()

yelp['text length'] = yelp['text'].apply(len)

def plot_data(yelp):
	sns.set_style('white')
	get_ipython().run_line_magic('matplotlib', 'inline')
	sns.show()

	g = sns.FacetGrid(yelp,col='stars')
	g.map(plt.hist,'text length')
	sns.show()

	sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
	sns.show()

	sns.countplot(x='stars',data=yelp,palette='rainbow')
	sns.show()

	stars = yelp.groupby('stars').mean()
	stars.corr().show()

	sns.heatmap(stars.corr(),cmap='coolwarm',annot=True).show()

#plot_data(yelp)

stars = yelp.groupby('stars').mean()

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]

X = yelp_class['text']
y = yelp_class['stars']

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

nb = MultinomialNB()
nb.fit(X_train,y_train)
predictions = nb.predict(X_test)

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)

predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))