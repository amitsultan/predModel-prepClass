import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model
import warnings
warnings.simplefilter("ignore")

from sklearn.metrics import accuracy_score
df_train = pd.read_csv("prep_matches.csv")
df_test = pd.read_csv("prep_predict.csv")


RF_clf = RandomForestClassifier(n_estimators = 200, random_state = 1, class_weight = 'balanced')
AB_clf = AdaBoostClassifier(n_estimators = 200, random_state = 2)
GNB_clf = GaussianNB()
KNN_clf =  KNeighborsClassifier()
LOG_clf = linear_model.LogisticRegression(multi_class = "ovr", solver = "sag", class_weight = 'balanced')
GRAD_clf = GradientBoostingClassifier(random_state=0)
EST_clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')

clfs = [GRAD_clf, LOG_clf, RF_clf, AB_clf, GNB_clf, KNN_clf]
from sklearn.metrics import mean_squared_error
y_train = df_train.pop('status')
X_train = df_train
y_test = df_test.pop('status')
X_test = df_test

EST_clf.fit(X_train, y_train)

for model in clfs:
    model.fit(X_train, y_train)
    expected = y_test
    predicted = model.predict(X_test)
    # print("Score of {} for training set: {:.4f}.".format(model.__class__.__name__, accuracy_score(y_train, model.predict(X_train))))
    print("Score of {} for test set: {:.4f}.".format(model.__class__.__name__, accuracy_score(y_test, model.predict(X_test))))
    # summarize the fit of the model
    # print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))
