# example of random forest classifier (thanks internet person)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data/ad.data', header=None)

response_var = df[len(df.columns.values)-1]
df.drop(df.columns[[len(df.columns)-1]],axis=1,inplace=True)

y = [1 if e == 'ad.' else 0 for e in response_var]
X = df

# deal with missing values
X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)


aucs = []

trees = [2,5,10,25,500]

for tree in trees:
    rf = RandomForestClassifier(n_estimators = tree,n_jobs=2)

    rf.fit(X_train,y_train)
    predrf = rf.predict(X_test)

    #print pd.crosstab(pd.core.series.Series(y_test), predrf, rownames=['actual'], colnames=['preds'])
    
    scores = cross_val_score(rf, X_train, y_train, scoring='roc_auc',cv=5)
    print(tree,"aucs: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    
    aucs.append(roc_auc_score(y_test,predrf))
    #print roc_auc_score(y_test,predrf)


plt.plot(trees,aucs)
