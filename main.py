# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score


# Đọc dữ liệu từ file xls
xlsx_file = 'Concrete_Data.xlsx'
df_data = pd.read_excel(xlsx_file, header=0, usecols=['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water',
                                                      'Superficialize', 'Coarse Aggregate', 'Fine Aggregate', 'Age'])

df_data.head()
X = df_data.iloc[1:1029, :8]
data = pd.read_excel(xlsx_file, header=0, usecols=[
                     'Concrete compressive strength'])
data.head()
Y = data.iloc[1:1029, :1]

# tách tập dữ liệu thành 70:30
X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, random_state=0, test_size=0.3)

one = np.ones((X_train.shape[0], 1))
Xbar = np.concatenate((one, X_train, ), axis=1)

regr = linear_model.LinearRegression(fit_intercept=False).fit(Xbar, y_train)


A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y_train)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]
y1 = w_0 + Xbar[:, 1]*w_1 + Xbar[:, 2]*w_2 + Xbar[:, 3]*w_3 + Xbar[:, 4] * \
    w_4 + Xbar[:, 5]*w_5 + Xbar[:, 6]*w_6 + Xbar[:, 7]*w_7 + Xbar[:, 8]*w_8
# print('Du doan bang cong thuc:', y1)
# print('Solution found by (5): ', w.T)

# fit_intercept = False for calculating the bias
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y_train)

# print('--------------------------------------------')

# print("Kết quả theo thư viện scikit-learn : ")
# print("W = ", regr.coef_)
# print("Kết quả dự đoán theo thư viện scikit-learn: ",
#       regr.predict(Xbar).reshape(-1))

# print('--------------Error theo thuật toán---------------')

# Tách tập dữ liệu làm 4 phần
A_train, A_test, b_train, b_test = train_test_split(
    X_train, y_train, random_state=0, test_size=0.5)

C_train, C_test, d_train, d_test = train_test_split(
    A_train, b_train, random_state=0, test_size=0.5)

E_train, E_test, f_train, f_test = train_test_split(
    A_test, b_test, random_state=0, test_size=0.5)

# Lan 1
G_train = np.concatenate((C_train, C_test, E_train))
G_test = np.concatenate((d_train, d_test, f_train))

# train error
print("\nTheo thuat toan lan 1")
one = np.ones((G_train.shape[0], 1))
Gbar = np.concatenate((one, G_train, ), axis=1)
A = np.dot(Gbar.T, Gbar)
b = np.dot(Gbar.T, G_test)
w = np.dot(np.linalg.pinv(A), b)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]
y_pred = w_0 + Gbar[:, 1]*w_1 + Gbar[:, 2]*w_2 + Gbar[:, 3]*w_3 + Gbar[:, 4] * \
    w_4 + Gbar[:, 5]*w_5 + Gbar[:, 6] * \
    w_6 + Gbar[:, 7]*w_7 + Gbar[:, 8]*w_8
mse = metrics.r2_score(G_test, y_pred)
print("Train error lan 1:", mse)

# validtion
one = np.ones((E_test.shape[0], 1))
Ebar = np.concatenate((one, E_test, ), axis=1)
A = np.dot(Ebar.T, Ebar)
b = np.dot(Ebar.T, f_test)
w = np.dot(np.linalg.pinv(A), b)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]
y2 = w_0 + Ebar[:, 1]*w_1 + Ebar[:, 2]*w_2 + Ebar[:, 3]*w_3 + Ebar[:, 4] * \
    w_4 + Ebar[:, 5]*w_5 + Ebar[:, 6] * \
    w_6 + Ebar[:, 7]*w_7 + Ebar[:, 8]*w_8
mse = metrics.r2_score(f_test, y2)
print("Validation error lan 1:", mse)

print("Theo thu vien lan 1")
# train error
one = np.ones((G_train.shape[0], 1))
Gbar = np.concatenate((one, G_train, ), axis=1)
y_pred = regr.predict(Gbar)
mse = metrics.r2_score(G_test, y_pred)
print("Train error lan 1:", mse)

# validation
one = np.ones((E_test.shape[0], 1))
Ebar = np.concatenate((one, E_test, ), axis=1)
y_pred = regr.predict(Ebar)
mse = metrics.r2_score(f_test, y_pred)
print("Validation error:", mse)

# -------------------------------- Lan 2 -------------------------------------
H_train = np.concatenate((E_test, C_test, E_train))
H_test = np.concatenate((f_test, d_test, f_train))

# train error
print("\nTheo thuat toan lan 2")
one = np.ones((H_train.shape[0], 1))
Hbar = np.concatenate((one, H_train, ), axis=1)
A = np.dot(Hbar.T, Hbar)
b = np.dot(Hbar.T, H_test)
w = np.dot(np.linalg.pinv(A), b)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]
y_pred = w_0 + Hbar[:, 1]*w_1 + Hbar[:, 2]*w_2 + Hbar[:, 3]*w_3 + Hbar[:, 4] * \
    w_4 + Hbar[:, 5]*w_5 + Hbar[:, 6] * \
    w_6 + Hbar[:, 7]*w_7 + Hbar[:, 8]*w_8
mse = metrics.r2_score(H_test, y_pred)
print("Train error lan 2:", mse)

# validation
one = np.ones((C_train.shape[0], 1))
Hbar = np.concatenate((one, C_train, ), axis=1)
A = np.dot(Hbar.T, Hbar)
b = np.dot(Hbar.T, d_train)
w = np.dot(np.linalg.pinv(A), b)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]
y2 = w_0 + Hbar[:, 1]*w_1 + Hbar[:, 2]*w_2 + Hbar[:, 3]*w_3 + Hbar[:, 4] * \
    w_4 + Hbar[:, 5]*w_5 + Hbar[:, 6] * \
    w_6 + Hbar[:, 7]*w_7 + Hbar[:, 8]*w_8
mse = metrics.r2_score(d_train, y2)
print("Validation error lan 2:", mse)

print("Theo thu vien lan 2")
# train error
one = np.ones((H_train.shape[0], 1))
Gbar = np.concatenate((one, H_train, ), axis=1)
y_pred = regr.predict(Gbar)
mse = metrics.r2_score(H_test, y_pred)
print("Train error lan 2:", mse)

# validation
one = np.ones((C_train.shape[0], 1))
Ebar = np.concatenate((one, C_train, ), axis=1)
y_pred = regr.predict(Ebar)
mse = metrics.r2_score(d_train, y_pred)
print("Validation error lan 2:", mse)

# ------------------------------------------------Lan 3---------------------------------------
# Lan 3
F_train = np.concatenate((E_test, E_train, C_train))
F_Test = np.concatenate((f_train, f_test, d_train))

print("\nTheo thuat toan lan 3")
one = np.ones((F_train.shape[0], 1))
Fbar = np.concatenate((one, F_train, ), axis=1)

A = np.dot(Fbar.T, Fbar)
b = np.dot(Fbar.T, F_Test)
w = np.dot(np.linalg.pinv(A), b)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]

y_pred = w_0 + Fbar[:, 1]*w_1 + Fbar[:, 2]*w_2 + Fbar[:, 3]*w_3 + Fbar[:, 4] * \
    w_4 + Fbar[:, 5]*w_5 + Fbar[:, 6] * \
    w_6 + Fbar[:, 7]*w_7 + Fbar[:, 8]*w_8
mse = metrics.r2_score(F_Test, y_pred)
print("Train error lan 3:", mse)
# validtion
one = np.ones((C_test.shape[0], 1))
Cbar = np.concatenate((one, C_test, ), axis=1)
A = np.dot(Cbar.T, Cbar)
b = np.dot(Cbar.T, d_test)
w = np.dot(np.linalg.pinv(A), b)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]
y2 = w_0 + Cbar[:, 1]*w_1 + Cbar[:, 2]*w_2 + Cbar[:, 3]*w_3 + Cbar[:, 4] * \
    w_4 + Cbar[:, 5]*w_5 + Cbar[:, 6] * \
    w_6 + Cbar[:, 7]*w_7 + Cbar[:, 8]*w_8
mse = metrics.r2_score(d_test, y2)
print("Validation error lan 3:", mse)

print("Theo thu vien lan 3")
# train error
one = np.ones((F_train.shape[0], 1))
Fbar = np.concatenate((one, F_train, ), axis=1)
y_pred = regr.predict(Fbar)
mse = metrics.r2_score(F_Test, y_pred)
print("Train error lan 3:", mse)
# validation
one = np.ones((C_test.shape[0], 1))
Cbar = np.concatenate((one, C_test, ), axis=1)
y_pred = regr.predict(Cbar)
mse = metrics.r2_score(d_test, y_pred)
print("Validation error lan 3:", mse)

# ----------------------------------------Lan 4-------------------------------------
CE_train = np.concatenate((E_test, C_test, C_train))
CE_Test = np.concatenate((f_test, d_test, d_train))

print("\nTheo thuat toan lan 4")
one = np.ones((CE_train.shape[0], 1))
CEbar = np.concatenate((one, CE_train, ), axis=1)

A = np.dot(CEbar.T, CEbar)
b = np.dot(CEbar.T, CE_Test)
w = np.dot(np.linalg.pinv(A), b)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]

y_pred = w_0 + CEbar[:, 1]*w_1 + CEbar[:, 2]*w_2 + CEbar[:, 3]*w_3 + CEbar[:, 4] * \
    w_4 + CEbar[:, 5]*w_5 + CEbar[:, 6] * \
    w_6 + CEbar[:, 7]*w_7 + CEbar[:, 8]*w_8
mse = metrics.r2_score(CE_Test, y_pred)
print("Train error lan 4:", mse)
# validtion
one = np.ones((E_train.shape[0], 1))
Exbar = np.concatenate((one, E_train, ), axis=1)
A = np.dot(Exbar.T, Exbar)
b = np.dot(Exbar.T, f_train)
w = np.dot(np.linalg.pinv(A), b)
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]
w_3 = w[3][0]
w_4 = w[4][0]
w_5 = w[5][0]
w_6 = w[6][0]
w_7 = w[7][0]
w_8 = w[8][0]
y2 = w_0 + Exbar[:, 1]*w_1 + Exbar[:, 2]*w_2 + Exbar[:, 3]*w_3 + Exbar[:, 4] * \
    w_4 + Exbar[:, 5]*w_5 + Exbar[:, 6] * \
    w_6 + Exbar[:, 7]*w_7 + Exbar[:, 8]*w_8
mse = metrics.r2_score(f_train, y2)
print("Validation error lan 4:", mse)

print("Theo thu vien lan 4")
# train errors
one = np.ones((CE_train.shape[0], 1))
Fbar = np.concatenate((one, CE_train, ), axis=1)
y_pred = regr.predict(Fbar)
mse = metrics.r2_score(CE_Test, y_pred)
print("Train error lan 4:", mse)
# validation
one = np.ones((E_train.shape[0], 1))
Cbar = np.concatenate((one, E_train, ), axis=1)
y_pred = regr.predict(Cbar)
mse = metrics.r2_score(f_train, y_pred)
print("Validation error lan 4:", mse)

# test error
one = np.ones((X_test.shape[0], 1))
Xbar = np.concatenate((one, X_test, ), axis=1)
y_pred = w_0 + Xbar[:, 1]*w_1 + Xbar[:, 2]*w_2 + Xbar[:, 3]*w_3 + Xbar[:, 4] * \
    w_4 + Xbar[:, 5]*w_5 + Xbar[:, 6]*w_6 + Xbar[:, 7]*w_7 + Xbar[:, 8]*w_8
mse = metrics.r2_score(y_test, y_pred)
print("\nTest error theo thuat toan:", mse)

# Test
one = np.ones((X_test.shape[0], 1))
Xbar = np.concatenate((one, X_test, ), axis=1)
y_pred = regr.predict(Xbar)

# test error
mse = metrics.r2_score(y_test, y_pred)
print("Test error theo thu vien:", mse)

# Đánh giá
regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)
print("Danh gia Linear Regression: ",
      explained_variance_score(y_test, y_pred))


