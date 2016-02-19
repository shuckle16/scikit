# example of random forest classifier (thanks internet person)

import pandas as pd
import matplotlib.pyplot as plt
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

aucs = []
neighbors = [2,3,4,5,8,10,12,25]

for neighbs in neighbors:
    knn = KNeighborsClassifier(neighbs, weights='distance')
    knn.fit(X_train, y_train)

    predknn = knn.predict(X_test)

    #print pd.crosstab(pd.core.series.Series(y_test), predknn, rownames=['actual'], colnames=['preds'])
    scores = cross_val_score(knn, X_train, y_train, scoring='roc_auc',cv=5)
    print(neighbs,"Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    aucs.append(roc_auc_score(y_test,predknn))
    #print neighbs, roc_auc_score(y_test,predknn)


plt.plot(neighbors,aucs)
