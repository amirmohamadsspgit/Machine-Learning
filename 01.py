import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
import cv2 as cv

T1 = []
T2 = []
T3 = []
E1 = []
E2 = []
E3 = []
S1 = []
S2 = []
S3 = []
for i in [10, 100, 1000, 10000 ]:
    X, y = make_blobs(n_samples=i, n_features=3, random_state=15, cluster_std=10)
    # ------------------------------------KNeighborsClassifier--------------------------
    tm1 = time.time()
    model_k = KNeighborsClassifier(n_neighbors=2)
    model_k.fit(X, y)
    y_p_k = model_k.predict(X)
    ttm1 = time.time()
    # --------------------------------------------------------------
    tm2 = time.time()
    model_l = LogisticRegression()
    model_l.fit(X, y)
    y_p_l = model_l.predict(X)
    ttm2 = time.time()
    # --------------------------------------------------------------
    tm3 = time.time()
    model_s = SVC()
    model_s.fit(X, y)
    y_p_s = model_s.predict(X)
    ttm3 = time.time()
    # --------------------------------------------------------------
    tim_k = ttm1 - tm1
    T1.append(tim_k)
    tim_l = ttm2 - tm2
    T2.append(tim_l)
    tim_s = ttm3 - tm3
    T3.append(tim_s)
    # --------------------------------------------------------------
    error_k = mean_squared_error(y, y_p_k)
    E1.append(error_k)
    error_l = mean_squared_error(y, y_p_l)
    E2.append(error_l)
    error_s = mean_squared_error(y, y_p_s)
    E3.append(error_s)
    # --------------------------------------------------------------
    score_k = model_k.score(X, y)
    S1.append(score_k)
    score_l = model_l.score(X, y)
    S2.append(score_l)
    score_s = model_s.score(X, y)
    S3.append(score_s)
# --------------------------------------------------------------
plt.subplot(3, 3, 1)
plt.plot(["10", "100", "1000", "10000" ] , T1)
plt.ylabel("TIME")
plt.title("KNN")
plt.subplot(3, 3, 4)
plt.plot(["10", "100", "1000", "10000" ] , S1)
plt.ylabel("SCORE")
plt.subplot(3, 3, 7)
plt.plot(["10", "100", "1000", "10000" ] , E1)
plt.ylabel("ERROR")
# --------------------------------------------------------------
plt.subplot(3, 3, 2)
plt.plot(["10", "100", "1000", "10000" ] , T2)
plt.title("LogisticRegression")
plt.subplot(3, 3, 5)
plt.plot(["10", "100", "1000", "10000" ] , S2)
plt.subplot(3, 3, 8)
plt.plot(["10", "100", "1000", "10000" ] , E2)
# --------------------------------------------------------------
plt.subplot(3, 3, 3)
plt.plot(["10", "100", "1000", "10000" ] , T3)
plt.title("SVC")
plt.subplot(3, 3, 6)
plt.plot(["10", "100", "1000", "10000" ] , S3)
plt.subplot(3, 3, 9)
plt.plot(["10", "100", "1000", "10000" ] , E3)
# --------------------------------------------------------------
plt.show()

