# stochastic gradient descent with grid search for best parameters

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier


df = pd.read_csv('data/ad.data', header=None)
explanatory_vars = set(df.columns.values)
response_var = df[len(df.columns.values)-1]
explanatory_vars.remove(len(df.columns.values)-1)

y = [1 if e == 'ad.' else 0 for e in response_var]
X = df[list(explanatory_vars)]

# deal with missing values
X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)


params = [{'alpha': [.1,.01,.001,.0001],'n_iter':[100,500,1000]}]

grid_searcher = GridSearchCV(SGDClassifier(), params, n_jobs=1, verbose=1, cv=3, scoring='roc_auc')
grid_searcher.fit(X_train, y_train)
print('Best score: %0.3f' % grid_searcher.best_score_)

best_parameters = grid_searcher.best_estimator_.get_params()

final_model = SGDClassifier(alpha=best_parameters['alpha'],n_iter=best_parameters['n_iter']).fit(X_train,y_train)

predictions = final_model.predict(X_test)

print(classification_report(y_test, predictions))

print pd.crosstab(pd.core.series.Series(y_test), predictions, rownames=['actual'], colnames=['preds'])

print roc_auc_score(y_test,predictions)
