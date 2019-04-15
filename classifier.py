import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

import graphviz 

dataset = np.genfromtxt('tseries_1,2_all_7_5_t.csv', delimiter=',')

np.random.shuffle(dataset)

split_length = 1200

X_train,y_train = dataset[:split_length,:-1],dataset[:split_length,-1]
X_test, y_test = dataset[split_length:,:-1],dataset[split_length:,-1]

print("Train shape (Rows,datapoints):",X_train.shape)
print("Test shape: (Rows,datapoints)",X_test.shape)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=10, random_state=0).fit(X_train, y_train)

print("Accuracy Score (GBT): ",clf.score(X_test, y_test))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print("Accuracy Score (DTree): ",clf.score(X_test, y_test))


# features = [str(x) for x in range(56)]
# classNames = ["Good","Bad"]

# dot_data = tree.export_graphviz(clf, 
# 	out_file="decisiontree.dot",
# 	feature_names=features,
# 	class_names=classNames,
# 	filled=True)


# clf = GaussianNB()
# clf = clf.fit(X_train, y_train)
# print("Accuracy Score (NB): ",clf.score(X_test, y_test))



