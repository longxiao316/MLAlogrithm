#encoding=utf-8
from classification.svm import svm1
import numpy as np
x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = svm1.SVM()
clf.fit(x,y,2)
print(clf.predict([[-0.8, -1]]))