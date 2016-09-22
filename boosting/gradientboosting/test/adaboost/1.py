# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import boosting.gradientboosting.adaBoost.Adaboost1 as Adaboost

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

x1, y1 = make_gaussian_quantiles(cov=2, n_samples=200, n_features=2, n_classes=2, random_state=1)
x2, y2 = make_gaussian_quantiles(mean=(3,3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1)
x = np.concatenate((x1, x2))
y = np.concatenate((y1, -y2 + 1))
y[y==0]=-1


clf=Adaboost.Adaboost()
clf.train(x,y,500)

plot_colors = "br"
plot_step = 0.02
classnames = "AB"

plt.figure(figsize=(10, 10))


xmin, xmax = x[:, 0].min() - 1, x[:, 0].max() + 1
ymin, ymax = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xmin, xmax, plot_step), np.arange(ymin, ymax, plot_step))

z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
plt.contourf(xx,yy,z,cmap=plt.cm.Paired)

# Plot the training points
for i, n, c in zip([-1,1], classnames, plot_colors):
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1],
                c=c, cmap=plt.cm.Paired,
                label="Class %s" % n)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

plt.show()