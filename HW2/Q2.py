import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import math
def Rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2
def dRosenbrock(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),200*(x[1]-x[0]**2)])
def Himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
def dHimmelblau(x):
    df_value=[2*(x[0]**2+x[1]-11)*2*x[0]+2*(x[0]+x[1]**2-7),2*(x[0]**2+x[1]-11)+2*(x[0]+x[1]**2-7)*2*x[1]]
    return np.array(df_value)


def plot_rosenbrock():
    # 生成数据
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = Rosenbrock([X, Y])
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(X, Y, Z,cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    # 显示图形
    plt.show()
def plot_himmelblau():
    # 生成数据
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = Himmelblau([X, Y])
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(X, Y, Z,cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    # 显示图形
    plt.show()

def plot_():
    plot_rosenbrock()
    plot_himmelblau()


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





def gradient_descent(x0,f,df,rho1,rho2,alpha0,tol,max_iter):
    iter_time=0
    X=[x0]
    while True:
        if np.linalg.norm(df(x0))<tol or iter_time>max_iter:
            break
        iter_time+=1
        d=-df(x0)
        #将d归一化
        d=d/np.linalg.norm(d)
        alpha,_=linear_research(x0,f,df,d,alpha0,rho1,rho2)
        x0=x0+alpha*d
        X.append(x0)
    return x0,iter_time,X


def sci_optim():
    x0 = [0,0]
    res = minimize(Rosenbrock, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
    print(res.x)
    print(res.fun)
    # 利用内置函数求解 himmelblau 函数的最小值
    x0 = [0,0]
    res = minimize(Himmelblau, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
    print(res.x)


def find_f_min(x0,f,df,rho1,rho2,alpha0,tol,max_iter):
    # 求解Rosenbrock函数的最小值
    start = time.time()
    x,iter_time,_ = gradient_descent(x0,f,df,rho1,rho2,alpha0,tol,max_iter)
    #取出f的函数名
    f_name = f.__name__
    print(f_name,"函数的最小值为：",f(x))
    print("选取的初值点：",x0)
    print("最小值点为：",x)
    print("迭代次数为：",iter_time)
    end = time.time()
    print("运行时间：",end-start)

def main():
    # 求解Rosenbrock函数的最小值
    x0 = np.array([0,0])
    tol = 1e-8
    max_iter = 2000
    rho1 = 0.2
    rho2 = 0.8
    alpha0 = 0.3
    # # find_f_min(x0,Rosenbrock,dRosenbrock,rho1,rho2,alpha0,tol,max_iter)
    # #不同初值求解
    # for i in range(25):
    #     x0=np.random.uniform(-10,10,size=(2,))
    #     try:
    #         find_f_min(x0,Rosenbrock,dRosenbrock,rho1,rho2,alpha0,tol,max_iter)
    #     except Exception as e:
    #         print("----")
    #         print(e)
    #         print("初始点为：",x0)
    #         print("----")
    #         continue
    #计算收敛阶
    max_iter = 5000
    _,_,X=gradient_descent(x0,Rosenbrock,dRosenbrock,rho1,rho2,alpha0,tol,max_iter)
    x_star=np.array([1,1])
    q=math.log(np.linalg.norm(X[-1]-x_star)/np.linalg.norm(X[-2]-x_star))/math.log(np.linalg.norm(X[-2]-x_star)/np.linalg.norm(X[-3]-x_star))
    print("该算法的收敛阶为：",q)
    #内置函数求解Rosenbrock函数的最小值
    sci_optim()



    # 求解Himmelblau函数的最小值
    # x0 = np.array([1,1])
    # tol = 1e-8
    # max_iter = 100
    # rho1 = 0.3
    # rho2 = 0.7
    # alpha0 = 1.5
    # find_f_min(x0,Himmelblau,dHimmelblau,rho1,rho2,alpha0,tol,max_iter)
    # x0 = np.array([2,-1])
    # find_f_min(x0,Himmelblau,dHimmelblau,rho1,rho2,alpha0,tol,max_iter)
    # x0= np.array([-1,2])
    # find_f_min(x0,Himmelblau,dHimmelblau,rho1,rho2,alpha0,tol,max_iter)
    # x0= np.array([-1,-2])
    # find_f_min(x0,Himmelblau,dHimmelblau,rho1,rho2,alpha0,tol,max_iter)
    # # 寻找初始值
    tol = 1e-8
    max_iter = 2000
    rho1 = 0.2
    rho2 = 0.8
    alpha0 = 0.3
    for i in range(30):
        x0 = np.random.uniform(0,10,size=(2,))
        try:
          find_f_min(x0,Himmelblau,dHimmelblau,rho1,rho2,alpha0,tol,max_iter)
        except Exception as e:
            print("----")
            print(e)
            print("初始点为：",x0)
            print("----")
            continue
if __name__ == '__main__':
    main()
    # plot_()