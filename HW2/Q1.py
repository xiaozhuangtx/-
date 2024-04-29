import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
#插值法确定满足wolfe 条件的 alpha
def f(x):
    return x[0]**4-4*x[0]**3+6*x[0]**2-4*x[0]+1
def df(x):
    df_value=[4*x[0]**3-12*x[0]**2+12*x[0]-4]
    return np.array(df_value)

def plot_one_var():
    x=np.arange(-100,100,0.1)
    y=f(x)
    plt.plot(x,y)
    plt.xlabel("x")
    plt.yticks( )
    plt.ylabel("f(x)")
    plt.title(r'$ f(x) = 3x^4-16x^3+30x^2-24x+8 $')
    plt.show()

def Wolfe_Condition(x0,f,df,alpha,d,rho1,rho2):
    if rho1>1/2 or rho2<=rho1 or rho2>=1:
    #抛出异常
        raise Exception("Invalid rho1 or rho2")
    if f(x0+alpha*d) > f(x0) + rho1*alpha*(df(x0)@d):
        flag = 1
    elif df(x0+d*alpha)@d < rho2*(df(x0)@d):
        flag = 2
    else:
        flag = 0
    return flag

def linear_research(x0,f,df,d,alpha0,rho1,rho2):
    iter_time = 0
    alpha=alpha0
    while True:
        iter_time+=1
        if iter_time >= 50:
            raise Exception("插值陷入死循环")
        if Wolfe_Condition(x0,f,df,alpha0,d,rho1,rho2) == 0:
            break
        elif Wolfe_Condition(x0,f,df,alpha0,d,rho1,rho2) == 1:
            alpha0 = -1/2*alpha0**2*(df(x0)@d)/(f(x0+alpha0*d)-f(x0)-alpha0*(df(x0)@d))
        else:
            alpha0 = -(df(x0)@d)*alpha0/(df(x0+alpha0*d)@d-(df(x0)@d))
    return alpha0,iter_time


def super_params_analysis1():
    X=[]
    Iter_time_list=[]
    Alpha=[]
    for i in range(1000):
        x0=np.random.uniform(-100,100,size=(1,))
        X.append(x0)
        d=-df(x0)
        alpha0=1
        alpha, iter_time = linear_research(x0,f,df,d,alpha0,0.2,0.8)
        Alpha.append(alpha)
        Iter_time_list.append(iter_time)
    plt.scatter(X,Iter_time_list,1)
    plt.xlabel("x0")
    plt.ylabel("iter_time")
    plt.title("iter_time-x0")
    plt.show()
    plt.figure()
    plt.scatter(X,Alpha,1)
    plt.xlabel("x0")
    plt.ylabel("alpha")
    plt.title("alpha-x0")
    plt.show()
def super_params_analysis2():
    Alpha=[]
    rho_list=[]
    Iter_time=[]
    for rho1 in np.linspace(0.01, 0.5-0.01, 100):
        for rho2 in np.linspace(rho1+0.05, 1-0.01, 100):
            rho_list.append([rho1,rho2])
            x0=np.random.uniform(-10,10,size=(1,))
            d=-df(x0)
            alpha0=1
            alpha, iter_time = linear_research(x0,f,df,d,alpha0,rho1,rho2)
            Alpha.append(alpha)
            Iter_time.append(iter_time)
    #画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([i[0] for i in rho_list], [i[1] for i in rho_list], Iter_time)
    ax.set_xlabel('rho1')
    ax.set_ylabel('rho2')
    ax.set_zlabel('iter_time')
    plt.show()
    #找出最小的iter_time对应的rho1和rho2
    min_iter_time=min(Iter_time)
    index=Iter_time.index(min_iter_time)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter([i[0] for i in rho_list], [i[1] for i in rho_list],Alpha)
    ax.set_xlabel('rho1')
    ax.set_ylabel('rho2')
    ax.set_zlabel('alpha')
    plt.show()

def super_params_analysis3():
    Iter_time=[]
    Alpha=[]
    for i in range(1000):
        alpha0=np.random.uniform(0,100)
        Alpha.append(alpha0)
        x0=np.random.uniform(-10,10,size=(1,))
        d=-df(x0)
        alpha, iter_time = linear_research(x0,f,df,d,alpha0,0.2,0.8)
        Iter_time.append(iter_time)
    plt.scatter(Alpha,Iter_time,1)
    plt.xlabel("alpha0")
    plt.ylabel("iter_time")
    plt.title("iter_time-alpha0")
    plt.show()

def gradient_descent(x0,f,df,rho1,rho2,alpha0,tol,max_iter):
    iter_time=0
    while True:
        if np.linalg.norm(df(x0))<tol or iter_time>max_iter:
            break
        iter_time+=1
        d=-df(x0)
        #将d归一化
        d=d/np.linalg.norm(d)
        alpha,_=linear_research(x0,f,df,d,alpha0,rho1,rho2)
        x0=x0+alpha*d
    return x0,iter_time
def one_var_func_min():
    start = time.time()
    x0 = np.random.uniform(-10,10,size=(1,))
    tol = 1e-8
    max_iter = 1000
    x,iter_time = gradient_descent(x0,f,df,tol,max_iter)
    print("一元函数的最小值为：",f(x))
    print("最小值点为：",x)
    print("迭代次数为：",iter_time)
    end = time.time()
    print("运行时间：",end-start)

#插值的英文：interpolation
def Test_inter_func():
    # 测试插值函数
    alpha0 = 0.5
    x0=np.random.uniform(-10,10,size=(1,))
    d=-df(x0)
    rho1=0.2
    rho2=0.8
    alpha,iter_time=linear_research(x0,f,df,d,alpha0,rho1,rho2)
    print("alpha0=",alpha0)
    print("x0=",x0)
    print("iter_time=",iter_time)
    print("alpha=",alpha)




def find_f_min(x0,f,df,rho1,rho2,alpha0,tol,max_iter):
    # 求解Rosenbrock函数的最小值
    start = time.time()

    x,iter_time = gradient_descent(x0,f,df,rho1,rho2,alpha0,tol,max_iter)
    print("Rosenbrock函数的最小值为：",f(x))
    print("选取的初值点：",x0)
    print("最小值点为：",x)
    print("迭代次数为：",iter_time)
    end = time.time()
    print("运行时间：",end-start)


# plot_one_var()

#插值
Test_inter_func()

#参数分析
# super_params_analysis1()
# super_params_analysis2()
# super_params_analysis3()

