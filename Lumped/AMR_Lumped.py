import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
from matplotlib import rc
rc('text',usetex = True)
rc('font', family='serif',size = 12)


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import CoolProp.CoolProp as Cool



def Length(x): 
    """
    Length(Length)
    
    Returns a correction for the effective magnetic field regarding length of the magnetic circuit
    """
    
    return(1.008766163651124 +(-0.5079975035831273/4)*x + (2.6682392253711824/12)*x**2 + (-6.636314340353568/32)*x**3)
    
def Prop_param(X):
    """
    C_H,C_L,dT_mg,dT_dmg = Prop_param(T_c,T_h,B)
    
    Returns the thermomagnetic properties of the magnetocaloric material
    """
    
    T_c,T_h,B = X
    C_H = 2.880197*T_c-2.012589*T_h-414.045137*B**2+1074.942047*B
    C_L = 3.534705*T_c-1.661617*T_h-44.487789*B**2+172.791961*B

    dT_mg   = 0.007788*T_c-0.007578*T_h-0.033219*B**2+1.644276*B
    dT_dmg = 0.008638*T_c-0.011177*T_h-1.414265*B**2+4.830623*B
        
    return(C_H,C_L,dT_mg,dT_dmg)
    
Effect = pd.read_excel('Data/Effect.xlsx', header = None)  # Pre-calculateed effectiveness
phi = [0.01,0.02,0.05,0.075,0.1,0.125,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.65] #Utilization factor
NTU = [10,25,50,100,150,250] #Number of Transfer uUnits

NN,PP = np.meshgrid(NTU,phi)
Effectiv = interpolate.interp2d(phi,NTU,Effect.T,kind = 'linear') #Function for the calculation of the effectivness of teh regenerator

def fit_w(X,a,b):
    """
    a*P1**b = fit_w(X,a,b)
    
    P1 = X
    
    Function to be used as the magnetizatio power correlation
    """
    
    P1= X
    
    return a*P1**b

def fit_int(X,a,b,c,d,e,f): 
    """
    a*P1+b*P2+c*P3+d*P4+e*P5+f*P6 = fit_int(X,a,b,c,d,e,f)
    
    P1,P2,P3,P4,P5,P6= X
    
    Function to be used as the Cooling Capacity and heat Rejection Rate correlations
    """
    
    P1,P2,P3,P4,P5,P6= X
    
    
    return (a*P1+b*P2+c*P3+d*P4+e*P5+f*P6)

def Lumped_Met(X):
    
    """
    Q_pas_CB_, Q_act_CB_, Q_pas_HB_, Q_act_HB_, Q_span_, Q_pump_, W_mag_, Q_cond_, m_s*f*K_, Ut_L, Ut_H = Lumped_Met(X)
    
    W, H,f,m_f,B_ap,L,T_h,T_c = X
    
    Return the metrics for the calculation of the Cooling Capacity and Magnetization Power by the Lumped Model
    """
    
    rho_s = 7000  #[kg/m3] Density of the solid
    k_s   = 8     #[W/(m2.K)] #Thermal conductivity of the solid

    Epsilon = 0.45      # Volumetric Posority  
    Epsilon_hyd = 0.307 # Equivalent Hydraulic Porosity 
    Epsilon_ht = 0.34   # Equivalent Heat Transfer Porosity 
    d_p = 650*10**(-6)  # Particle diameter
    
    W, H,f,m_f,B_ap,L,T_h,T_c = X
    
    T_h = T_h + 273.15
    T_c = T_c + 273.15
    m_f = m_f/3600
    # Properties of the Fluid
    
    
    mu_f=Cool.PropsSI('V','T',(T_c+T_h)/2,'P',101325,'Water')       #Viscosity of the Water [Pa.s]
    rho_f=Cool.PropsSI('D','T',(T_c+T_h)/2,'P',101325,'Water')        #Density  of the Water [kg/m3]
    k_f=Cool.PropsSI('L','T',(T_c+T_h)/2,'P',101325,'Water')        #Conductivity of the Water [W/(m.K)]
    Pr_f=Cool.PropsSI('PRANDTL','T',(T_c+T_h)/2,'P',101325,'Water') #Prandtl Number of the Water [-]
    cp_f=Cool.PropsSI('C','T',(T_c+T_h)/2,'P',101325,'Water')       #Specific Heat of the Water [kJ/(kg.K)]
    

    
    # Closure Relations
    FE = 0.375                                                # Blow fraction of the Regenerator
    m_s   = L*W*H*10**-9*(1-Epsilon)*rho_s                    # Mass of the selod phase of the regenerator [kg]
    Beta = (Epsilon_hyd/Epsilon_ht)*(1-Epsilon_ht)*6/(d_p)    # Surface area density of the regenerator (m2/m3)
    v_s   = (m_f)/(rho_f*W*H*10**-6)                          #Superficial velocity of the fluid flow [-]
    Re_dp = d_p*v_s*rho_f/(mu_f) #Reynolds number
    C_f = m_f*cp_f*FE #heat capacity of the liquid stream [kW/(kg.K)]

    d_P = (150*(1-Epsilon_hyd)**2/Epsilon_hyd**3*mu_f*v_s/d_p**2+1.75*(1-Epsilon_hyd)/Epsilon_hyd**3*rho_f*v_s**2/d_p)*L/1000 #Pressure drop of the regenerator [Pa]

    Nu = 2*(1+4*(1-Epsilon_hyd)/Epsilon_hyd)+((1-Epsilon_hyd)**0.5)*Re_dp**0.6*Pr_f**(1/3) # Nusselt number [-]
    h_1 = Nu*k_f/d_p       # Non-corrected convective heat transfer coeficient [W/K]
    Bi = h_1*(d_p/2)/(k_s) # Biot Number [-]

    k_e_f = k_f*Epsilon #Equivalent thermal conductivity of the fluid phase [W/(m.K)]

    a_0 = np.exp(-1.084-6.778*(Epsilon-0.298)) #Correction coefficient
    f_0 = 0.8                                  #Correction coefficient

    k_e_s = k_f*((1-a_0)*(Epsilon*f_0+(1-Epsilon*f_0)*k_s/k_f)/(1-Epsilon*(1-f_0)+k_s/k_f*Epsilon*(1-f_0))+a_0*(2*((k_s/k_f)**2)*(1-Epsilon)+(1+2*Epsilon)*k_s/k_f)/((2+Epsilon)*k_s/k_f+(1-Epsilon))) #Equivalent solid heat conductivity [W/(m.K)]
    Pe = Re_dp*Pr_f                  # Peclet Number [-]
    D_ = (k_f/rho_f/cp_f)*0.75*Pe/2  # Dispersion [m2/s]

    k_s_eff = k_e_s                  # Effective thermal conductivity of the solid phase [W/(m.K)]
    k_f_eff = k_e_f + rho_f*cp_f*D_  # Effective thermal conductivity of the fluid phase [W/(m.K)]
    
    #Calculation of the Effective Magnetic Field
    
    

    a = L/2 #[mm]
    b = W/2 #[mm]
    c = H/2 #[mm]
    
    ## Calculation of the Demagnetization Factor
    
    Nd = 1/np.pi*( (b**2-c**2)/(2*b*c)*np.log(((a**2+b**2+c**2)**0.5-a)/((a**2+b**2+c**2)**0.5+a)) + (a**2-c**2)/(2*a*c)*np.log(((a**2+b**2+c**2)**0.5-b)/((a**2+b**2+c**2)**0.5+b)) +b/(2*c)*np.log(((a**2+b**2)**0.5+a)/((a**2+b**2)**0.5-a)) + a/(2*c)*np.log(((a**2+b**2)**0.5+b)/((a**2+b**2)**0.5-b)) +  c/(2*a)*np.log(((c**2+b**2)**0.5-b)/((c**2+b**2)**0.5+b)) +  c/(2*b)*np.log(((c**2+a**2)**0.5-a)/((c**2+a**2)**0.5+a)) +2*np.arctan((a*b)/(c*(a**2+b**2+c**2)**0.5)) +(a**3 + b**3 - 2*c**3)/(3*a*b*c) +  (a**2+b**2-2*c**2)/(3*a*b*c)*(a**2+b**2+c**2)**0.5+c/(a*b)*((a**2+c**2)**0.5 + (b**2+c**2)**0.5)-((a**2 + b**2)**1.5 + (a**2+c**2)**1.5 + (b**2+c**2)**1.5)/(3*a*b*c))
    Nd_bed = 1/3 + (1-Epsilon)*(Nd-1/3)

    Fac = Length(L/189) # Correction for the length of the Magnetic Circuit
    Corr = 0.78         # Correction for the peak of the magnetic profile

    B = B_ap*Corr*Fac*(1-Nd_bed*0.081281) - Nd_bed*(0.001257*T_c-0.000352*T_h) # Effective Magnetic Field [T]
    
    
    
    #Calculation of the Thermo-Magnetic Properties
    
    C_H,C_L,dT_mg,dT_dmg = Prop_param((T_c,T_h,B)) # High field specific heat capacity [kJ/(kg.K)], low field specific heat capacity [kJ/(kg.K)], magnetization temperature change [K], demagnetization temperature change [K]
    
    #Correction on the thermal coefficient
    
    Fo_H = k_s/(rho_s*2*C_H*f*(d_p/2)**2) # Fourier number in the high magnetic field [-]
    Fo_L = k_s/(rho_s*2*C_L*f*(d_p/2)**2) # Fourier number in the low magnetic field [-]

    X_H = Fo_H*np.exp(0.246196 - 0.84878*np.log(Fo_H) - 0.05639*(np.log(Fo_H))**2) #Correction factor in the high magnetic field [-]
    X_L = Fo_L*np.exp(0.246196 - 0.84878*np.log(Fo_L) - 0.05639*(np.log(Fo_L))**2) #Correction factor in the low magnetic field [-]

    DF_H = 1/(1+2*Bi/5*X_H) # Degradation factor in the high magnetic field [-]
    DF_L = 1/(1+2*Bi/5*X_L) # Degradation factor in the low magnetic field [-]

    h_int_H = Nu*k_f/d_p*DF_H #Corrected heat transfer coefficient in the high magnetic field [W/(m.K)] 
    h_int_L = Nu*k_f/d_p*DF_L #Corrected heat transfer coefficient in the low magnetic field [W/(m.K)]

    Ut_L = (C_f)/(m_s*C_L*f) #Utilization factor in the low magnetic field region
    Ut_H = (C_f)/(m_s*C_H*f) #Utilization factor in the low magnetic field region

    

    NTU_H  = h_int_H*Beta*(L*W*H*10**(-9))/(C_f) #Correted Number of Transfer units of the high magnetic field period 
    NTU_L  = h_int_L*Beta*(L*W*H*10**(-9))/(C_f) #Correted Number of Transfer units of the low magnetic field period

    R_HB = (rho_f/rho_s)*(cp_f/C_L)*(Epsilon/(1-Epsilon)) #Corretion parameter
    R_CB = (rho_f/rho_s)*(cp_f/C_H)*(Epsilon/(1-Epsilon)) #Corretion parameter

    F_HB = 1 + 1.764*R_HB+1.0064*R_HB**2 #Corretion Coefficient
    F_CB = 1 + 1.764*R_HB+1.0064*R_HB**2 #Corretion Coefficient

    try:
        Efness_HB = np.zeros(len(Ut_L))
        Efness_CB = np.zeros(len(Ut_H))

        for i in range(len(Ut_L)):
            Efness_HB[i] = Effectiv(Ut_L[i],NTU_L[i]*F_HB[i])
            Efness_CB[i] = Effectiv(Ut_H[i],NTU_H[i]*F_CB[i])
    except:
            Efness_HB = Effectiv(Ut_L,NTU_L*F_HB)
            Efness_CB = Effectiv(Ut_H,NTU_H*F_CB)
        
    phi_L = (m_s*C_L*f)/(m_s*C_L*f+ m_f*cp_f) #Ratio of the heat capacities
    phi_H = (m_s*C_H*f)/(m_s*C_H*f+ m_f*cp_f) #Ratio of the heat capacities
    
    #Calculation of the Parameters of the Model
    
    Q_pas_CB_ = Efness_CB*(T_h - T_c)*m_f*cp_f*FE   #Energy associated with the passive operation of the regenerator during the Cold Blow
    Q_act_CB_ = dT_mg*m_f*cp_f*FE                   #Energy associated with the active operation of the regenerator during the Cold Blow
    Q_pas_HB_ = Efness_HB*(T_h - T_c)*m_f*cp_f*FE   #Energy associated with the passive operation of the regenerator during the Hot Blow
    Q_act_HB_ = dT_dmg*m_f*cp_f*FE                  #Energy associated with the active operation of the regenerator during the Hot Blow
    Q_span_ = (T_h - T_c)*m_f*cp_f*FE               #Energy associated with the temperature span of the regenerator
    Q_pump_ = d_P*m_f*2*FE/1000                     #Energy associated with the viscous dissipation of the regenerator
    W_mag_  = m_s*f/(T_h - T_c)*(C_H*dT_mg*phi_H - C_L*dT_dmg*phi_L) #Energy associated with the magnetization power of the regenerator
    Q_cond_ = ((1-Epsilon)*k_s_eff + Epsilon*k_f_eff)*(W*H*10**-6)*(T_h-T_c)/(L*10**-3)  #Energy associated with the heat conduction of the regenerator
    K_ = (C_H-C_L)*(T_h - T_c) #Correction factor for the magnetization power
    
    return Q_pas_CB_, Q_act_CB_, Q_pas_HB_, Q_act_HB_, Q_span_, Q_pump_, W_mag_, Q_cond_, m_s*f*K_, Ut_L, Ut_H


def Lumped_Model(X):
    """
    Lumped_Wm, Lumped_Qc = Lumped_Model(X)
    
    W, H,f,m_f,B_ap,L,T_h,T_c = X
    
    Returns the values of the magnetization power and of the cooling capacity according to the Lumped Model Calculations 
    
    """
    
    
    W, H,f,m_f,B_ap,L,T_h,T_c = X
    
    Coef_w = np.array([0.01395081, 0.85111948])
    Coef_c = np.array([ 1.7382991 ,  0.91797132, -1.70275788, -2.66603783, -3.98412347, -8.44748643])
    print()
    
    Q_pas_CB, Q_act_CB, Q_pas_HB, Q_act_HB, Q_span, Q_pump, W_mag, Q_cond, K, Ut_L, Ut_H =Lumped_Met(np.array(X).T)
    
    Lumped_Wm = fit_w(Ut_H,Coef_w[0],Coef_w[1])*K
    Lumped_Qc = fit_int((Q_pas_CB, Q_act_CB, Q_span, Lumped_Wm, Q_pump, Q_cond), 
                   Coef_c[0], Coef_c[1], Coef_c[2], Coef_c[3], Coef_c[4], Coef_c[5])
    
    return Lumped_Wm, Lumped_Qc[0]

