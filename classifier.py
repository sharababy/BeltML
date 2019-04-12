import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

dataset = np.genfromtxt('tseries_all_7_5_t.csv', delimiter=',')


X_train,y_train = dataset[:648,:-1],dataset[:648,-1]
X_test, y_test = dataset[648:,:-1],dataset[648:,-1]

print("Train shape (Rows,datapoints):",X_train.shape)
print("Test shape: (Rows,datapoints)",X_test.shape)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=10, random_state=0).fit(X_train, y_train)

print("Accuracy Score (GBT): ",clf.score(X_test, y_test))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print("Accuracy Score (DTree): ",clf.score(X_test, y_test))


clf = GaussianNB()
clf = clf.fit(X_train, y_train)
print("Accuracy Score (NB): ",clf.score(X_test, y_test))



