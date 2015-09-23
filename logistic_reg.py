from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

import pandas as pd

df = pd.read_csv('winequality-white.csv', header=0, sep=';')

X = df[list(df.columns)[:-1]]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
model.score(X_test, y_test)
