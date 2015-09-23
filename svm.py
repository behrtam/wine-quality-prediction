from sklearn import svm
from sklearn.cross_validation import train_test_split
import pandas as pd

df = pd.read_csv('dataset/winequality-white.csv', header=0, sep=';')

X = df[list(df.columns)[:-1]]
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=0)
model = svm.SVC(gamma=0.001, C=100.)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print model.score(X_test, y_test)
