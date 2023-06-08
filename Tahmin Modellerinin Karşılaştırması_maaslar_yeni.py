# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:27:10 2023

@author: 90539
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar_yeni.csv')
print(veriler)

#maası tahmin edeceğimiz için maas =y , geri kalanlar x olaca
#data mı buna göre ayırıyorum


x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values

#korelasyon matrisidir. 
print(veriler.corr())

#eğitim test
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.33, random_state = 0)

#çoklu doğrusal bağlantı

from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X,Y) #x_trainden y_traini öğreniyor

y_pred = lr.predict(X) #öğrenimden sonra x_teste bakarak y_testi tahmin edecek
print(y_pred)

#çoklu doğrusal bağlantı hata scoru hesaplaması
from sklearn.metrics import r2_score
print( ' Lineer Regresyon R2 Değeri')
print(r2_score(Y, y_pred))


#bazı değişkenleri çıkartmalıyız Backward İşlemi

import statsmodels.api as sm
model= sm.OLS(y_pred, X)
print(model.fit().summary())

#polinomal regresyon

from sklearn.preprocessing import PolynomialFeatures
P = PolynomialFeatures( degree =2) #ikinci dereceden polinomal regresyon oluştur demek
x_poly = P.fit_transform(X)
print(x_poly)

l2 = LinearRegression()
l2.fit(x_poly, y)


import statsmodels.api as sm
model2= sm.OLS(l2.predict(P.fit_transform(X)), X)
print(model2.fit().summary())




#SVR regresyon
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()

x_olcekli = sc1.fit_transform(X)


sc2 = StandardScaler()

y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR
svr= SVR(kernel= 'rbf')
svr.fit(x_olcekli, y_olcekli)


from sklearn.metrics import r2_score
print( ' SVR R2 Değeri')
print(r2_score(y_olcekli, svr.predict(x_olcekli)))


import statsmodels.api as sm
model3= sm.OLS(svr.predict(x_olcekli), x_olcekli)
print(model3.fit().summary())




#KARAR AĞACI REGRESYON İÇİN

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)


from sklearn.metrics import r2_score
print( ' DT R2 Değeri')
print(r2_score(Y, r_dt.predict(X)))


import statsmodels.api as sm
model4= sm.OLS(r_dt.predict(X), X)
print(model4.fit().summary())

#Random Forest

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10 , random_state=0) #10 adet farklı karar ağacı çizecek
rf_reg.fit(X,Y.ravel())


from sklearn.metrics import r2_score
print( ' Random Forsest R2 Değeri')
print(r2_score(Y, rf_reg.predict(X)))


import statsmodels.api as sm
model= sm.OLS(rf_reg.predict(X), X)
print(model.fit().summary())







































