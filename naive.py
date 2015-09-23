from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
import pandas as pd

df = pd.read_csv('dataset/winequality-white.csv', header=0, sep=';')

X = df[list(df.columns)[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)

modelg = GaussianNB()
modelg.fit(X_train, y_train)
y_predict = modelg.predict(X_test)
print "Gaus " + str(modelg.score(X_test, y_test))

modelm = MultinomialNB()
modelm.fit(X_train, y_train)
y_predict = modelm.predict(X_test)
print "Multi " + str(modelm.score(X_test, y_test))

modelb = BernoulliNB()
modelb.fit(X_train, y_train)
y_predict = modelb.predict(X_test)
print "Bernoulli " + str(modelb.score(X_test, y_test))