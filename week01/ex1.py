import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
练习一：单变量线性回归
"""

#预加载数据
path='ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])

#查看数据前五行
print(data.head())

#查看数据统计信息数据
print(data.describe())

#绘制散点图
data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))
plt.xlabel('Population (10,000s)')
plt.ylabel('Profit (10,000s)')
plt.show()

#实现代价函数
def computeCost(X,y,theta):
    """
    计算线性回归的代价函数
    参数：
        X: 特征矩阵
        y: 目标向量
        theta: 参数向量
    返回：
        代价函数的值
    """
    inner=np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))

#添加偏置项
data.insert(0,'Ones',1)

#提取特征矩阵X和目标向量y
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

#观察X和y是否正确
print(X.head())
print(y.head())

#转换为numpy矩阵
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))

print(computeCost(X,y,theta))

#梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    """
    批量梯度下降算法
    参数：
        X: 特征矩阵
        y: 目标向量
        theta: 参数向量
        alpha: 学习率
        iters: 迭代次数
    返回：
        theta: 最终的参数向量
        cost: 代价函数的值
    """
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    for i in range(iters):
        error=X*theta.T-y

        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j]-((alpha/len(X))*np.sum(term))
        
        theta=temp
        cost[i]=computeCost(X,y,theta)

    return theta,cost

alpha=0.01
iters=1000
g,cost=gradientDescent(X,y,theta,alpha,iters)
print(g)


#绘制结果
x=np.linspace(data.Population.min(),data.Population.max(),100)
f=g[0,0]+g[0,1]*x

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

#绘制代价函数
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

"""
练习二：多变量线性回归
"""

#预加载数据
path='ex1data2.txt'
data2=pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])
print(data.head())

#特征归一化
data2=(data2-data2.mean())/data2.std()
print(data2.head())

#重复练习一进行线性回归训练
data2.insert(0,'Ones',1)

cols=data2.shape[1]
X2=data2.iloc[:,0:cols-1]
y2=data2.iloc[:,cols-1:cols]

X2=np.matrix(X2.values)
y2=np.matrix(y2.values)
theta2=np.matrix(np.array([0,0,0]))

g2,cost2=gradientDescent(X2,y2,theta2,alpha,iters)
print(computeCost(X2,y2,g2))

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost2,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

