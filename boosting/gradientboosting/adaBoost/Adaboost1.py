#encoding=utf-8
import numpy as np
import decisiontree.DecisionStump as ds

class Adaboost(object):
    '''
    gradient boost取指数损失可以推出ａｄａｂｏｏｓｔ
    '''
    def __init__(self):
        self.models=[]
        self.alphas=[]
    def predict(self,x):
        print self.models
        print self.alphas
        db=ds.DecisionStumpBuilder()
        y=np.zeros((np.shape(x)[0],1))
        for s in self.models:
            a=s['alpha']
            predict=a*db.predict(x,s)
            y+=predict
        y[y>0]=1
        y[y<=0]=-1
        return y
    def train(self,x,y,numit,shrinkage=1):
        '''
        x,数据
        y,标签，
        numit,迭代次数
        shrinkage
        '''
        m,n = np.shape(x)
        #初始化第一次权重
        wei=np.ones((m,1))/float(m)
        #下面迭代优化
        db=ds.DecisionStumpBuilder()
        for i in range(numit):
            stump,est,err=db.buildStump(x,y,wei,20)#stump
            if err>.5:
                print "error"
            alpha=float(0.5*np.log((1-err)/np.max(err,1e-20)))
            alpha=alpha*shrinkage
            #更新权重w(t+1)=w(t)*当前指数错误
            #指数错误
            experr=np.exp(np.multiply(-1*alpha*np.mat(y),est.T))
            wei=np.multiply(wei.reshape(-1,1),experr.reshape(-1,1))
            stump['alpha']=alpha
            self.alphas.append(alpha)
            self.models.append(stump)