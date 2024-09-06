import numpy as np
from scipy import constants as C
import pandas as pd

#Wm  = 0.28 #0.8 #3.12 #0.28165
#Wr  = 1e-16#1e-16 #3.1# 3.1
#Wv  = 1-(Wm+Wr) #
#Primero se obtienen las condiciones iniciales.

def Universe_R_DataFrame(H_0,Wm,Wr,Wv):
    H_0 = H_0*60*60*24*365*1e6/C.parsec #7.116564*(10**(-2)) #(Gyear)**-1 69.60 km/(Mpc*s)
    a    = 2E-3
    t_0  = 0.0
    t_p  = 0.0
    R_0  = 2e-5#2e-3
    R_0p = R_0
    Rf   = []
    tf   = []
    WT  = Wm+Wr+Wv
    Wk  = 1.-WT

    def f(x):
        return x*((Wr+(Wm*x)+(Wk*(x**2))+(Wv*(x**4)))**(-0.5))

    def Rp(x):
        return H_0*((Wr/(x**2))+(Wm/x)+(Wk)+(Wv*(x**2)))**(0.5)

    def dis(x):
        return Wr/(x**2)+(Wm/x)+Wk+Wv*(x**2)

    if Wk <0 :
        k = -1
        print("signo de curvatura 1, curvatura positiva (espacio esférico, cerrado)")
    elif Wk ==0:
        k = 0
        print("signo de curvatura 0, curvatura cero (espacio plano,abierto)")
    else:
        k = 1
        print("signo de curvatura -1, curvatura negativa (espacio hiperbólico, abierto)")


        #condiciones iniciales
    suma = 0.0
    dx2  =  1E-7  #1,000 sub intervalos de integración 
    for o in np.arange(a,R_0+a,2*dx2):
        I_interpol = f(o)+4*f(o+dx2)+f(o+2*dx2)
        suma = suma + I_interpol
        sumaf = (dx2/3.0)*suma
    t_0 = t_0 + sumaf/(H_0)
    tf.append(t_0)
    Rf.append(R_0)

    #Ahora se resuelve la ec por método de euler
    t_f     = 40.0 #tiempo final en Gyears 
    dx      = 1E-3 #mil puntos en la integral
    dot_R_0 = Rp(R_0)
    t       = np.arange(t_0,t_f,dx)
    tdis    =t_f

    for i in t :
#        print(dx)
        if i<tdis:
            y = Rp(R_0)*dx+R_0
            Rf.append(y)
            R_0  = y
            tf.append(i)
            if dis(R_0)<=(2e-7):
                tdis =i
                print("hola apertura",R_0,dis(R_0))
        else:
            y = -Rp(R_0)*dx+R_0
            Rf.append(y)
            tf.append(i)
            R_0  = y
            print(y)
            if y <= R_0p:
            	tdis =t_f
            	print("hola cierre")
#        dx = 1e-3*(f(R_0)/dot_R_0)+1E-9
    Rf = np.array(Rf)
    tf = np.array(tf)
    DataF_RU = pd.DataFrame({"time":tf,"radi":Rf})
    return DataF_RU

def Universe_Age(H_0,Wm,Wr,Wv):
    H_0 = H_0*60*60*24*365*1e6/C.parsec #7.116564*(10**(-2)) #(Gyear)**-1 69.60 km/(Mpc*s)
    a    = 2E-3
    t_0  = 0.0
    t_p  = 0.0
    R_0p = 2e-5#2e-3
    WT  = Wm+Wr+Wv
    Wk  = 1.-WT

    def f(x):
        return x*((Wr+(Wm*x)+(Wk*(x**2))+(Wv*(x**4)))**(-0.5))

    if Wk <0 :
        k = -1
        print("signo de curvatura 1, curvatura positiva (espacio esférico, cerrado)")
    elif Wk ==0:
        k = 0
        print("signo de curvatura 0, curvatura cero (espacio plano,abierto)")
    else:
        k = 1
        print("signo de curvatura -1, curvatura negativa (espacio hiperbólico, abierto)")

    #valores en el presente
    suma_p = 0.0
    dx2_p  =  1E-3  #1,000 sub intervalos de integración 
    for o in np.arange(a,1,2*dx2_p):
        I_interpol_p = f(o)+4*f(o+dx2_p)+f(o+2*dx2_p)
        suma_p = suma_p + I_interpol_p
        sumaf_p = (dx2_p/3.0)*suma_p
    t_p = t_p + sumaf_p/(H_0)
    return [t_p,1]
