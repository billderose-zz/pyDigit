import pandas
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


train = pandas.read_csv("train.csv")
test = pandas.read_csv("test.csv")
response = train.icol(0)
predictors = train.loc[:, "pixel0":]

rf = RandomForestClassifier(n_estimators = 400, bootstrap = True, 
                            n_jobs = -1, verbose = True)
erf = ExtraTreesClassifier(n_estimators = 400, bootstrap = True, 
                           n_jobs = -1, verbose = True)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2),
                         n_estimators = 500,
                         learning_rate = 1)

rf.fit(predictors, response)
erf.fit(predictors, response)
ada.fit(predictors, response)
rf_pred = pandas.Series(rf.predict(test))
erf_predict = pandas.Series(erf.predict(test))
ada_predict = pandas.Series(ada.predict(test))

# start indexing at 1 to comply with competition rules
rf_pred.index += 1
erf_predict.index += 1
ada_predict.index += 1

rf_pred.to_csv("rf.csv")
erf_pred.to_csv("erf.csv")
ada_pred.to_csv("ada.csv")



