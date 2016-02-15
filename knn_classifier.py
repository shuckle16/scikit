import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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

knn = KNeighborsClassifier(50, weights='distance')
knn.fit(X_train, y_train)

predknn = knn.predict(X_test)

print pd.crosstab(pd.core.series.Series(y_test), predknn, rownames=['actual'], colnames=['preds'])

print roc_auc_score(y_test,predknn)
