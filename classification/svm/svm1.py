#encoding=utf-8
import numpy as np
import random as rd

class SVM(object):
    def __init__(self,k=(lambda a,b:a*b),lin=True):
        '''
        k:核函数
        '''
        self.k=k
        self.lin=lin
    # def selectI(self):#选择第一个变量
    #
    #     pass
    def selectJRandom(self,i,m):#
        j=i
        while j==i:
            j=int(rd.uniform(0,m))
        return j
    def selectJ(self,x,y,a,b):#选择第二个变量使得Ej-Ei最大

        pass

    def fit(self,x,y,c):
        '''

        x:数据
        y:标签
        c:线性不可分的惩罚系数
        '''
        k=self.k
        m,n=np.shape(x)
        #算E
        self.ecache=np.zeros((m,2))
        #先通过smo算出a
        #选择两个优化alpha
        a=np.zeros(m)
        b=0
        #下面迭代优化
        for i in range(5000):
            #选两个优化的alpha
            for i in range(m):
                if i>0 and i<c:#TODO heuristic ??????
                    # i=self.selectI(a)
                    j=self.selectJRandom(i,m)#self.selectJ(a,i)
                    eta=k(a[i],a[i])+k(a[j],a[j])-2*k(a[i],a[j])

                    #计算ei，ej

                    ei=np.dot(a,np.multiply(y,k(x,x[i])))+b-y[i]
                    ej=np.dot(a,np.multiply(y,k(x,x[i])))+b-y[j]
                    #更新ai
                    aiold=a[i].copy()
                    ajold=a[j].copy()
                    ainew=a[i]+y[i]*(ej-ei)/eta
                    #计算a的L,H
                    if y[i]*y[j]<=0:
                        L=np.max(0,np.abs(a[i]-a[j]),0)
                        H=np.min(c,np.abs(c-a[i]+a[j]))
                    else:
                        L=np.max(0,np.abs(a[i]+a[j]-c))
                        H=np.min(c,np.abs(a[i]+a[j]))
                    if ainew>H:
                        ainew=H
                    if ainew<L:
                        ainew=L
                    #更新a[j]
                    ajnew=ajold+aiold-ainew
                    a[i]=ainew#这里可以判断是否有更新，如果n轮没有更新的话可以提前终止迭代,
                    a[j]=ajnew
                    #下面更新b，更新b的时候利用求出的e 简化运算
                    bj=y[j]*(ajold-ajnew)*k(x[j],x[j])+y[i](aiold-ainew)*k(x[i],x[j])-b-ej
                    bi=y[i]*(aiold-ainew)*k(x[i],x[i])+y[j][ajold-ajnew]*k(x[j],x[i])-b-ei
                    if c>ainew>0:
                        b=bi
                    elif c>ajnew>0:
                        b=bj
                    else:
                        b=(bi+bj)/2
        self.b=b
        #一轮迭代结束
        #发现用非线性核的话，这个w暂时我没有找到方法求，（如果线性可以根据w的偏导求），暂时存支撑向量
        if self.lin:
            self.w=np.dot(self.lin(x),np.multiply(a,y))
        else:
            self.a=a[a!=0]
            self.x=x[a!=0,:]
            self.y=y[a!=0]

    def predict(self,v):
        val=0
        if self.k:
           val= np.dot(self.a,np.multiply(self.y,self.k(self.x,v)))+self.b
        else:
            val=np.dot(self.w,v)+b
        return val