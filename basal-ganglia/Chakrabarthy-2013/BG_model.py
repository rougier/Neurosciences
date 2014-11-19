# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Distributed under the (new) BSD License.
# Copyright (c) 2014, Nicolas P. Rougier
# 
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr),
#               Meropi Topalidou (Meropi.Topalidou@inria.fr)



# Packages import



from dana import *
import matplotlib.pyplot as plt
#%pylab --no-import-all


# Helper functions



def H(x):
    return 1 if x > 0 else 0
   

# Simulation parameters

# Population size
n = 20

# Default Time resolution
dt = 1.0*millisecond

# Default trial duration
duration = 100.0*dt

# Initialization of the random generator (reproductibility !)
np.random.seed(1)


# Dopamine levels
delta = -2

# Time constants
tau_GPe = 10
tau_STN = 10
tau_GPi = 10

theta_D1 =0.1
theta_D2 = 0.1
theta_GPe = 0.1
theta_STN = 0.1


# Compute of parameters
lamda_D1 = 5*(1/(1+np.exp(-6*(delta - theta_D1))))
lamda_D2 = 5*(1/(1+np.exp(-6*(theta_D2 - delta))))
lamda_GPe = 4 * H(theta_GPe - delta) + 1;  
lamda_STN = 4 * H(theta_STN - delta) + 1

r = np.zeros((n,n,n,n))
for i in range(0,n):
    for j in range(0,n):
        for p in range(0,n):
            for q in range(0,n):
                r[i,j,p,q] = np.sqrt((i-p)**2 + (j-q)**2)
                      
d = np.zeros((n,n))
for i in range(0,n):
    for p in range(0,n):
        d[i,p] = np.abs(i-p)

                           


k_x = 2*np.pi/n                                
A = 10 
sigma =1.2
C = 0.2
#parameters A, sigma, Ce{0.1,0.3}, k_x, tau are from the article from Standage 


W_lat = 1
#W_lat = sigma_STN * np.exp(-r^2/sigma_lat^2 ) if r < R else -1 if r =0 else 0; r; R; sigma_STN = 1; sigma_lat = 0.2; 
#w_sg = 1
#w_gs = 1
#W_GPe = np.ones((1,n))
W_GPi = A * np.exp(-d**2/(2*sigma**2)) - C;
#W_STN_GPi = np.ones((1,n))*1./n

# Populations

Striatum_D1 = zeros((n,1), """du/dt = -u + V + I_ext;    
                              V = np.tanh(lamda_D1* u); 
                              I_ext""")
                                             
Striatum_D2 = zeros((n,1), """du/dt = -u + V + I_ext;
                              V = np.tanh(lamda_D2* u);   
                              I_ext""") 

GPe = zeros((n,n), """dx/dt = (-x +  W_lat * np.ones(U.shape) * np.sum(U) +  I_STN + I_Str) / tau_GPe;
                      U = np.tanh(lamda_GPe * x); 
                      I_STN; I_Str""")
                      #I_Str = W_GPe * V_D2; 
                                 
                                
STN = zeros((n,n), """dx/dt = (-x +  W_lat * np.ones(U.shape) * np.sum(U) +  I_GPe) / tau_STN;    
                      U = np.tanh(lamda_STN * x); I_GPe""")
                                

GPi = zeros((n,1), """du/dt = (-u + np.dot(W_GPi, S) * k_x + I) / tau_GPi;  
                                S = u**2 / (1 + 1/2 * k_x * np.sum(u**2)); 
                                I = V_D1  +  U_STN; U_STN ; V_D1   """)

# Connectivity


DenseConnection( Striatum_D2('V'), GPe('I_Str'), np.ones((1,n)) )
DenseConnection( STN('U'), GPe('I_STN'), 1.0 )
DenseConnection( GPe('U'), STN('I_GPe'), 1.0 )
DenseConnection( Striatum_D1('V'), GPi('V_D1'), 1.0 )
DenseConnection( STN('U'), GPi('U_STN'), np.ones((1,n))*1./n )

#Stimulus


@clock.at(1*millisecond)
def stimulus(time):
    sigma = 0.4 #sigmaE{0.3,0.5}
    d = np.linspace(0,n-1,n).reshape((n,1))
    I = np.zeros((n,1))
    first_start = 3
    first_num_neur = 4
    first = np.arange(first_start, first_start + first_num_neur)
    second_start = 13
    second_num_neur = 3
    second = np.arange(second_start, second_start + second_num_neur)
    I[first] = 2
    I[second] = 4
    Striatum_D1['I_ext'] = I
    Striatum_D2['I_ext'] = I


#Save GPi's activity in each tick


Gpi = np.zeros((int(duration*1000),n))
@after(clock.tick)
def GP_i(t):
    index = int(t*1000)
    Gpi[index,:] = GPi["S"].T

# Run simulation


run(time=duration, dt=dt)

#Choice

choice = 1 if np.sum(GPi("S")[0:10]) > np.sum(GPi("S")[10:20]) else 2
print choice
print "diff = " ,np.abs(np.sum(GPi("S")[0:10]) - np.sum(GPi("S")[10:20]))

#Plot of output of GPi

plt.plot(Gpi[:,2],"b")
plt.plot(Gpi[:,3],"b")
plt.plot(Gpi[:,12],"r")
plt.plot(Gpi[:,13],"r")

plt.show()

