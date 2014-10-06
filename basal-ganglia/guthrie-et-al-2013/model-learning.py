# **A Long Journey into Reproducible Computational Neurosciences Research**  
# Meropi Topalidou¹²³, Thomas Boraud³ and Nicolas P. Rougier¹²³*
# 
# ¹ INRIA Bordeaux Sud-Ouest, France  
# ² LaBRI, UMR 5800 CNRS, Talence, France  
# ³ Institute of Neurodegenerative Diseases, UMR 5293, Bordeaux, France  
# * Corresponding author ([Nicolas.Rougier@inria.fr](mailto:Nicolas.Rougier@inria.fr))

# Packages import

from dana import *
import matplotlib.pyplot as plt
import os
import time

# Simulation parameters

# Population size
n = 4

# Default trial duration
duration = 3.0*second

# Default Time resolution
dt = 1.0*millisecond

# Initialization of the random generator (reproductibility !)
np.random.seed(1)

# Threshold
Cortex_h   =  -3.0
Striatum_h =   0.0
STN_h      = -10.0
GPi_h      =  10.0
Thalamus_h = -40.0

# Time constants
Cortex_tau   = 0.01
Striatum_tau = 0.01
STN_tau      = 0.01
GPi_tau      = 0.01
Thalamus_tau = 0.01

# Noise leve (%)
Cortex_N   =   0.01
Striatum_N =   0.001
STN_N      =   0.001
GPi_N      =   0.03
Thalamus_N =   0.001

# Sigmoid parameters
Vmin       =   0.0
Vmax       =  20.0
Vh         =  16.0
Vc         =   3.0

# Learning parameters
a_c = 0.05
a_aLTP = 0.002
a_aLTD= 0.001
Wmin = 0.25 
Wmax = 0.75

# Initialization of Values

simulation = 0
trial = 0

T = []
C_Cx, M_Cx, A_Cx = [], [], []
C_Str, M_Str, A_Str = [], [], []
C_STN, M_STN = [], []
C_GPi, M_GPi = [], []
C_Th, M_Th = [], []

Cue_Rw_Choice = np.zeros((120,7))
equal_m_c = np.zeros(120)

# Learning weights
Weights = np.zeros((121,4))
V_value = np.zeros((120,4))

Cue_Values = np.ones((4,2))*7.0
PR = np.linspace(0.,1.,4)
R = np.zeros(120)
Right_Choice = np.zeros(120)
V = 0.5 * np.ones((4,1))

# Helper functions

def Boltzmann(V,Vmin=Vmin,Vmax=Vmax,Vh=Vh,Vc=Vc):
    return  Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))


def noise(V, level):
    return  V + np.random.normal(0,(abs(V)+0.0001)*level,V.shape)

def init_weights(Initial_Weights, gain=1):
    global Wmin, Wmax 
    N = np.random.normal(0.5, 0.005, Initial_Weights.shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return gain*Initial_Weights*(Wmin+(Wmax-Wmin)*N)

def min_max(w, Wmin = Wmin, Wmax = Wmax):
    return np.minimum(Wmax,np.maximum(w[w!=0],Wmin))

# Populations

Cortex_cog   = zeros((n,1), """dV/dt = -V +U + (-V + noise(I+Iext,Cortex_N) - Cortex_h)/Cortex_tau;
                               U = np.maximum(V,0); I; Iext""")
Cortex_mot   = zeros((1,n), """dV/dt = -V +U + (-V +noise(I+Iext,Cortex_N) - Cortex_h)/Cortex_tau;
                               U = np.maximum(V,0); I; Iext""")
Cortex_ass   = zeros((n,n), """dV/dt = -V +U + (-V + Iext - Cortex_h)/Cortex_tau;
                               U = np.maximum(noise(V,Cortex_N),0); Iext""")
Striatum_cog = zeros((n,1), """dV/dt = -V +U + (-V + noise(Boltzmann(I), Striatum_N) - Striatum_h)/Striatum_tau;
                               U = np.maximum(V,0); I""")
Striatum_mot = zeros((1,n), """dV/dt = -V +U + (-V + noise(Boltzmann(I), Striatum_N) - Striatum_h)/Striatum_tau;
                               U = np.maximum(V,0); I""")
Striatum_ass = zeros((n,n), """dV/dt = -V +U + (-V + noise(Boltzmann(0.2*I + I_Ass), Striatum_N) - Striatum_h)/Striatum_tau;
                               U = np.maximum(V,0); I; I_Ass""")
STN_cog      = zeros((n,1), """dV/dt = -V +U + (-V + noise(I, STN_N) - STN_h)/STN_tau;
                               U = np.maximum(V,0); I""")
STN_mot      = zeros((1,n), """dV/dt = -V +U + (-V +  noise(I, STN_N) - STN_h)/STN_tau;
                               U = np.maximum(V,0); I""")
GPi_cog      = zeros((n,1), """dV/dt = -V +U + (-V + noise(-2.0*I_Str + I , GPi_N) - GPi_h)/GPi_tau;
                               U = np.maximum(V,0); I_Str; I""")
GPi_mot      = zeros((1,n), """dV/dt = -V +U + (-V + noise(-2.0*I_Str + I , GPi_N) - GPi_h)/GPi_tau;
                               U = np.maximum(V,0); I_Str; I""")
Thalamus_cog = zeros((n,1), """dV/dt = -V +U + (-V -0.5*I + 0.4*I_Cx - Thalamus_h)/Thalamus_tau;
                               U = np.maximum(noise(V, Thalamus_N),0); I; I_Cx""")
Thalamus_mot = zeros((1,n), """dV/dt = -V +U + (-V -0.5*I + 0.4*I_Cx - Thalamus_h)/Thalamus_tau;
                               U = np.maximum(noise(V,Thalamus_N),0); I; I_Cx""")
# Connectivity

Cog_Con = DenseConnection( Cortex_cog('V'),   Striatum_cog('I'), 1.0)
Initial_Cog_Con = Cog_Con._weights
Cog_Con._weights = init_weights(Initial_Cog_Con) 

Mot_Con = DenseConnection( Cortex_mot('V'),   Striatum_mot('I'), 1.0)
Initial_Mot_Con = Mot_Con._weights
Mot_Con._weights = init_weights(Mot_Con._weights)

Ass_Con = DenseConnection( Cortex_ass('V'),   Striatum_ass('I_Ass'), 1.0)
Initial_Ass_Con = Ass_Con._weights
Ass_Con._weights = init_weights(Ass_Con._weights)

Cog_Ass_Con = DenseConnection( Cortex_cog('V'),   Striatum_ass('I'), np.ones((1,2*n+1)))
Initial_Cog_Ass_Con = Cog_Ass_Con._weights
Cog_Ass_Con._weights = init_weights(Cog_Ass_Con._weights)

Mot_Ass_Con = DenseConnection( Cortex_mot('V'),   Striatum_ass('I'), np.ones((2*n+1,1)))
Initial_Mot_Ass_Con = Mot_Ass_Con._weights
Mot_Ass_Con._weights = init_weights(Mot_Ass_Con._weights)

DenseConnection( Cortex_cog('V'),   STN_cog('I'),       1.0 )
DenseConnection( Cortex_mot('V'),   STN_mot('I'),       1.0 )
DenseConnection( Striatum_cog('V'), GPi_cog('I_Str'),      1.0 )
DenseConnection( Striatum_mot('V'), GPi_mot('I_Str'),      1.0 )
DenseConnection( Striatum_ass('V'), GPi_cog('I_Str'),      np.ones((1,2*n+1)))
DenseConnection( Striatum_ass('V'), GPi_mot('I_Str'),      np.ones((2*n+1,1)))
DenseConnection( STN_cog('V'),      GPi_cog('I'),       1.0*np.ones((2*n+1,1)) )
DenseConnection( STN_mot('V'),      GPi_mot('I'),       1.0*np.ones((1,2*n+1)) )
DenseConnection( GPi_cog('V'),      Thalamus_cog('I'), 1.0 )
DenseConnection( GPi_mot('V'),      Thalamus_mot('I'), 1.0 )
DenseConnection( Thalamus_cog('V'), Cortex_cog('I'),    1.0 )
DenseConnection( Thalamus_mot('V'), Cortex_mot('I'),    1.0 )
DenseConnection( Cortex_cog('V'),   Thalamus_cog('I_Cx'),  1.0 )
DenseConnection( Cortex_mot('V'),   Thalamus_mot('I_Cx'),  1.0 )

# Cues Randomization for the Simulation

choices_cog  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
choices_mot  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])

cues_cog = choices_cog.copy()
cues_mot = choices_mot.copy()

for i in range(1,20):
    cues_cog = np.vstack((cues_cog,choices_cog))
    cues_mot = np.vstack((cues_mot,choices_mot))

# Record Ensembles' Activity

@after(clock.tick)
def register(t):
    T.append(t)
    C_Cx.append(Cortex_cog['V'].copy().ravel())
    M_Cx.append(Cortex_mot['V'].copy().ravel())
    A_Cx.append(Cortex_cog['V'].copy().ravel())
    C_Str.append(Cortex_cog['V'].copy().ravel())
    M_Str.append(Cortex_mot['V'].copy().ravel())
    A_Str.append(Cortex_mot['V'].copy().ravel())
    C_STN.append(Cortex_cog['V'].copy().ravel())
    M_STN.append(Cortex_mot['V'].copy().ravel())
    C_GPi.append(Cortex_cog['V'].copy().ravel())
    M_GPi.append(Cortex_mot['V'].copy().ravel())
    C_Th.append(Cortex_cog['V'].copy().ravel())
    M_Th.append(Cortex_mot['V'].copy().ravel())

# Trial setup


@clock.every(3.0*second,500*millisecond)
def set_trial(t):
    global cues_cog, cues_mot, trial, Cue_Values
    print 'trial = ', trial+1
    c1, c2 = cues_cog[trial,0], cues_cog[trial,1]
    m1, m2 = cues_mot[trial,0], cues_mot[trial,1]
    print 'cues_cog = ', c1,c2
    print 'cues_mot = ', m1,m2

    Cortex_cog['Iext'][c1,0]  = Cue_Values[c1,0]
    Cortex_cog['Iext'][c2,0]  = Cue_Values[c2,0]
    Cortex_mot['Iext'][0,m1]  = Cue_Values[m1,1]
    Cortex_mot['Iext'][0,m2]  = Cue_Values[m2,1]
    Cortex_ass['Iext'][c1,m1] = Cue_Values[c1,0]
    Cortex_ass['Iext'][c2,m2] = Cue_Values[c2,0]

# Selection

no_move = []
@clock.every(3.0*second, 2.999*second)
def selection(t):
	global trial, R, V
    
	if trial == 0:
		Weights[trial] = Cog_Con.weights[0,0], Cog_Con.weights[1,1], Cog_Con.weights[2,2], Cog_Con.weights[3,3]
		
	move = -np.sort(-Cortex_mot['V'])[:,0] - 40 > -np.sort(-Cortex_mot['V'])[:,1]
	print 'move = ', move
	if move:
		choice = np.argmax(Cortex_mot['V'])
		print 'choice = ', choice
		cognitive = np.argmax(Cortex_cog['V'])
		cog_right_ch = np.argmax(cues_cog[trial,:])
		mot_right_ch = cues_mot[trial,cog_right_ch]
        
		ch = np.where(choice == cues_mot[trial])
		print 'moto choice place = ', ch
		print 'cognitive = ', cognitive
		print 'cues_cog[trial,ch] = ', cues_cog[trial,ch]
		equal_m_c[trial] = (1 if cognitive == cues_cog[trial,ch] else 0)
		print 'equal_m_c = ', equal_m_c[trial]
		Right_Choice[trial] = (1 if mot_right_ch == choice else 0)
		print 'Right_Choice = ', Right_Choice[trial]
		if np.any(cues_mot[trial] == choice):
			R[trial] = (1 if np.random.random() < PR[choice] else 0)
		else:
			R[trial] = 0
		
		print 'Reward = ', R[trial], '\n\n'
		PE = R[trial] - V[choice]
		V[choice] = V[choice] + (PE * a_c)
		V_value[trial] = V.T
		a = (a_aLTP if PE>0 else a_aLTD)
		dw = PE  * a * Striatum_cog['V'][choice][0]
		
	
	
		Cog_Con.weights[choice, choice] = min_max(Cog_Con.weights[choice, choice] + dw)

		Weights[trial+1] = Cog_Con.weights[0,0], Cog_Con.weights[1,1], Cog_Con.weights[2,2], Cog_Con.weights[3,3]
        
	else:
		no_move.append([trial])
        Cue_Rw_Choice[trial, 4] =  1
	Cue_Rw_Choice[trial, 5:] =  R[trial], Right_Choice[trial]
	trial += 1

# Reset Of Activity Ensembles After Every Trial


@clock.every(3.0*second,3.0*second)
def reset(time):
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0
    
    Cortex_cog['V'] = 0
    Cortex_mot['V'] = 0
    Cortex_ass['V'] = 0

    Striatum_cog['V'] = 0
    Striatum_mot['V'] = 0
    Striatum_ass['V'] = 0

    STN_cog['V'] = 0
    STN_mot['V'] = 0

    GPi_cog['V'] = 0
    GPi_mot['V'] = 0

    Thalamus_cog['V'] = 0
    Thalamus_mot['V'] = 0

# Run simulation

#start = time.clock()
#new_trial = time.clock()
duration = 360*second
dt = 1*millisecond
run(time=duration, dt=dt)

# Display Cortex's Activity from a Trial    

np.save('C_Cx.npy',C_Cx)
np.save('M_Cx.npy',M_Cx)
trial_time = 3 * 1000 
duration = np.linspace(trial_time*0,trial_time*1-1,3000).astype(int)

C = np.array(C_Cx)
M = np.array(M_Cx)

plt.plot( C[duration],'b')
plt.plot(C[duration],'b')
plt.plot(C[duration],'b')
plt.plot(C[duration],'b')
plt.plot(M[duration],'r')
plt.plot(M[duration],'r')
plt.plot(M[duration],'r')
plt.plot(M[duration],'r')
plt.show()

# Display of Weights' Changing During Learning

np.save('Weights.npy',Weights)
plt.plot(Weights[:,0], 'y')
plt.plot(Weights[:,1], 'b')
plt.plot(Weights[:,2], 'r')
plt.plot(Weights[:,3], 'm')
plt.show()

# Display of Values' Changing During Learning

np.save('V_value.npy',V_value)
plt.plot(V_value[:,0], 'y')
plt.plot(V_value[:,1], 'b')
plt.plot(V_value[:,2], 'r')
plt.plot(V_value[:,3], 'm')
plt.show()

# Display Equality of Cognitive and Motor Choice

zeros = np.array(np.where(equal_m_c == 0.0))
print zeros.size

ones = np.array(np.where(equal_m_c == 1.0))


plt.hist(equal_m_c, bins = 3)
plt.show()

plt.plot(zeros,'*')
plt.show()

