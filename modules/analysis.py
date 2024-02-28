import numpy as np
import matplotlib.pylab as plt


def chem_pot(size, fol='spartian_1'):
    mu_A=[]
    s_A=[]
    mu_B=[]
    s_B=[]

    nu=7
    AT_size=10
    HY_size=5

    steps=10000


    data=open(fol+'/'+str(size)+'/Mean_Comp_Density.txt')
    for a in data.readlines():
        histo=int(a.split()[0])
        break
    #histo= 72

    data=np.genfromtxt(fol+'/'+str(size)+'/Mean_Comp_Density_Appended.txt').T
    x1=data[1]
    y1=data[2]*((x1>AT_size/2)*(x1<(AT_size/2+HY_size)))
    dx1=(x1[1]-x1[0])
    
    mu=[]
    for i in range(steps):
        y_inte=y1[i*histo:(i+1)*histo]
        mu.append(np.trapz(y_inte)*dx1)
        
    mu=np.array(mu)
    

    data=np.genfromtxt(fol+'/'+str(size)+'/Mean_Comp_Energy_AT.txt').T


    dl=1/(len(data[0]))
    lamb=np.arange(0,1,dl)
    graF = nu * lamb**( nu - 1 )
    dF1= dl*(graF*data[1]).sum()
    

    mu_A.append((mu+dF1)[mu!=0][-50:].mean())
    s_A.append((mu+dF1)[mu!=0][-50:].std())

    
    return mu[mu!=0],dF1,mu_A,s_A


def N_lines(size,rep=1):
    fil=open('IdSize/spartian_'+str(rep)+'/'+str(size)+'/Dens_A_spartian_'+str(rep)+'.profile')
    k=0
    for i,j in enumerate(fil.readlines()):
        if len(j.split())==2:
            #print(i,j[:-1])
            k+=1
        if k==2:
            return i-3
            
def Density(size,N=0,rep=1):
    num=N_lines(size)
    frame=num*N
    return np.genfromtxt('IdSize/spartian_'+str(rep)+'/'+str(size)+'/Dens_A_spartian_'+str(rep)+'.profile',skip_header=4+frame,max_rows=num-1).T

def dens(size,N=0,NN=100,rep=1):
    data=Density(size,N,rep)
    lines=N_lines(size,rep)
    
    
    x=[data[1][i*int(lines/NN):(i+1)*int(lines/NN)].mean() for i in range(NN)]
    y=[data[3][i*int(lines/NN):(i+1)*int(lines/NN)].mean() for i in range(NN)]
    dy=[data[3][i*int(lines/NN):(i+1)*int(lines/NN)].std() for i in range(NN)]
    

    return x,y,dy
