import numpy as np
class DecisionStumpBuilder(object):
    '''
    决策桩,
    '''
    def stumpClassify(self,data,dimen,thres,ineq):
        retarr=np.ones((np.shape(data)[0],1))
        if ineq=='lt':
            # retarr[data[dimen]<thres]=1
            retarr[data[:,dimen]<=thres]=-1
        else:
            retarr[data[:,dimen]>thres]=-1
        return retarr
    def calError(self,x,y,wei):
        err=np.abs(x-y)
        if(wei):
            weightederr=np.dot(wei,err)
            return weightederr
        else:
            return err
    def buildStump(self,x,y,step=10,wei=None):
        '''
        根据错误最小切分特征和切分值，
        如果用在adaboost中的时候，算错误的时候需要加权重wei
        x:数据
        y:数据的值
        step:搜索步数
        '''
        #遍历特征，然后按照一定步长((max-min)/step)在特征范围内搜索，选择错误最小的stump
        m,n=np.shape(x)
        minerr=np.inf
        beststump={}
        bestcalssest=np.zeros((m,1))
        #遍历特征
        for i in range(m):
            #第i个特征
            xi=x[m]
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


