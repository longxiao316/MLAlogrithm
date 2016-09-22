import decisiontree.DecisionStump as ds
import numpy as np
import matplotlib.pyplot as plt
t=ds.DecisionStumpBuilder()
x1=np.random.random(size=10)
x2=np.random.random(size=10)

y=np.array([1,1,1,1,1,-1,-1,-1,-1,-1])
wei=np.ones((10,1))
beststump,bestcalssest,minerr=t.buildStump(np.c_[x1,x2],y,wei)
xmax=np.max(x1)+1
xmin=np.min(x1)-1
ymax=np.max(x2)+1
ymin=np.min(x2)-1

plotstep=0.05
xx,yy=np.meshgrid(np.arange(xmin,xmax,plotstep),np.arange(ymin,ymax,plotstep))
z=t.predict(np.c_[xx.ravel(),yy.ravel()],beststump)
z=z.reshape(xx.shape)

plt.contourf(xx,yy,z,cmap=plt.cm.Paired)

# Plot the training points
for i, n, c in zip([-1,1], "AB", "br"):
    idx = np.where(y == i)
    plt.scatter(x1[idx], x2[idx],
                c=c, cmap=plt.cm.Paired,
                label="Class %s" % n)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')
plt.show()
