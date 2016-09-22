#encoding=utf-8
import numpy as np
from  classification.classifier import  Classifier
class DecisionStumpBuilder(Classifier):
    '''
    决策桩,
    '''
    def stumpClassify(self,data,dimen,thres,ineq):
        retarr=np.ones((np.shape(data)[0],1))
        if ineq=='lt':
            retarr[data[:,dimen]<=thres]=-1
        else:
            retarr[data[:,dimen]>thres]=-1

        return retarr
    def calError(self,x,y,wei):
        err=np.mat(np.ones((np.shape(x)[0],1)))
        y=y.reshape(-1,1)
        err[x==y]=0
        weightederr=err.T*wei
        return np.sum(weightederr)
    def predict(self,x,stump):
        threshold=stump['threshold']
        ine=stump['ineq']
        dim=stump['dim']
        _y=np.ones((np.shape(x)[0],1))
        if ine=='lt':
            _y[x[:,dim]<=threshold]=-1
        else:
            _y[x[:,dim]>threshold]=-1
        return _y
    def buildStump(self,x,y,wei=None,step=10,):
        '''
        根据错误最小切分特征和切分值，
        如果用在adaboost中的时候，算错误的时候需要加权重wei
        x:数据
        y:数据的值
        step:搜索步数
        return :
            stump:最佳决策桩信息
            est:样本上决策结果
            min：（加权后）错误数
        '''
        #遍历特征，然后按照一定步长((max-min)/step)在特征范围内搜索，选择错误最小的stump
        y=y.reshape(-1,1)
        m,n=np.shape(x)
        minerr=np.inf
        beststump={}
        bestcalssest=np.zeros((m,1))
        #遍历特征
        for i in range(n):
            #第i个特征
            xi=x[:,i]
            min=np.min(xi)
            max=np.max(xi)
            stepsize=(max-min)/step
            for j in range(-1,int(step)+1):
                thres=min+float(j)*stepsize
                for ineq in ['lt','gt']:
                    stumparr=self.stumpClassify(x,i,thres,ineq)
                    weierr=self.calError(stumparr,y,wei)
                    if weierr<minerr:
                        minerr=weierr
                        beststump['ineq']=ineq
                        beststump['threshold']=thres
                        beststump['dim']=i
                        bestcalssest=stumparr
        return beststump,bestcalssest,minerr


