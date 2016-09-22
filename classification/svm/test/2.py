import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf=SVC()
clf.fit(x,y)
print(clf.predict([[-0.8, -1]]))