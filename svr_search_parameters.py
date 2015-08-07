import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

#np.random.seed(11)

a = np.arange(100).reshape((100, 1)) + np.random.normal(3,2,1)
b = [5*np.sin(i) + i for i in range(100)]

x_train, x_test, y_train, y_test = train_test_split(a,b)

costs = [2**i for i in range(-5,10)]
gammas = [2**i for i in range(-15,5)]

mse_tests = []
best_mse = 100000
best_gamma = 0
best_c = 0
for c in costs:
    for g in gammas:
        svr_rbf = SVR(kernel='rbf', C=c, gamma=g)
        y_rbf = svr_rbf.fit(x_train, y_train).predict(x_test)
        tmp_mse = mean_squared_error(y_test,y_rbf)
        mse_tests.append(tmp_mse)
        if best_mse > tmp_mse:
            best_mse = tmp_mse
            best_gamma = g
            best_c = c
 

#plt.plot(mse_tests)
#plt.show()

svr_final = SVR(kernel='rbf',C=best_c,gamma=best_gamma)
y_final =  svr_final.fit(x_train,y_train).predict(x_test)

plt.plot(a,b,c='k',label='data')
plt.scatter(x_test,y_final,c='b',s=50)
plt.show()
