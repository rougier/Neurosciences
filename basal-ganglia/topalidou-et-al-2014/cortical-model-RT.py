# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors:
#  * Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#  * Meropi Topalidou (Meropi.Topalidou@inria.fr)
# -----------------------------------------------------------------------------

# Packages import
import sys
from dana import *
import matplotlib.pyplot as plt


# Simulation parameters
# -----------------------------------------------------------------------------
# Population size
n = 4

# Trial duration
duration = 3.0*second

# Default Time resolution
dt = 1.0*millisecond

# Initialization of the random generator
#  -> reproductibility
np.random.seed(1)


# Threshold
# -------------------------------------
Cortex_h   =  -3.0
Striatum_h =   0.0
STN_h      = -10.0
GPi_h      =  10.0
Thalamus_h = -40.0

# Time constants
# -------------------------------------
tau = 0.01
Cortex_tau   = tau #0.01#tau #
Striatum_tau = tau #0.01
STN_tau      = tau #0.01
GPi_tau      = tau #0.01
Thalamus_tau = tau #0.01
# Noise leve (%)
# -------------------------------------
Cortex_N   =   0.01
Striatum_N =   0.001
STN_N      =   0.001
GPi_N      =   0.03
Thalamus_N =   0.001

# Sigmoid parameters
# -------------------------------------
Vmin       =   0.0
Vmax       =  20.0
Vh         =  16.0
Vc         =   3.0

# Setup
# -------------------------------------
display  = True
cortical = True
gpi      = False
familiar = False

# With GPi (RT / 50 trials)
# familiar stimuli:   0.296 +/- 0.035
# unfamiliar stimuli: 0.477 +/- 0.138

# Without GPi (RT / 50 trials)
# familiar stimuli:   0.385 +/- 0.055
# unfamiliar stimuli: 0.718 +/- 0.222

# With GPi (RT / 120 trials)
# HC: 0.308 +/- 0.068
# NC: 0.486 +/- 0.132

# Without GPi (RT / 120 trials)
# HC: 0.399 +/- 0.055
# NC: 0.678 +/- 0.168



# Helper functions
# -----------------------------------------------------------------------------
def sigmoid(V,Vmin=Vmin,Vmax=Vmax,Vh=Vh,Vc=Vc):
    return  Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))

def noise(Z, level):
    Z = (1+np.random.uniform(-level/2,level/2,Z.shape))*Z
    return np.maximum(Z,0.0)

def init_weights(L, gain=1):
    Wmin, Wmax = 0.25, 0.75
    W = L._weights
    N = np.random.normal(0.5, 0.005, W.shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    L._weights = gain*W*(Wmin + (Wmax - Wmin)*N)

def reset():
    clock.reset()
    for group in network.__default_network__._groups:
        group['U'] = 0
        group['V'] = 0
        group['I'] = 0
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0


# Populations
# -----------------------------------------------------------------------------
Cortex_cog   = zeros((n,1), """dV/dt = (-V + I + Iext - Cortex_h)/(Cortex_tau);
                           U = noise(V,Cortex_N); I; Iext""")#min_max(V,-3.,60.)
Cortex_mot   = zeros((1,n), """dV/dt = (-V + I + Iext - Cortex_h)/(Cortex_tau);
                           U = noise(V,Cortex_N); I; Iext""")
Cortex_ass   = zeros((n,n), """dV/dt = (-V + I + Iext - Cortex_h)/(Cortex_tau);
                           U = noise(V,Cortex_N); I; Iext""")
Striatum_cog = zeros((n,1), """dV/dt = (-V + I - Striatum_h)/(Striatum_tau);
                           U = noise(sigmoid(V), Striatum_N); I""")
Striatum_mot = zeros((1,n), """dV/dt = (-V + I - Striatum_h)/(Striatum_tau);
                           U = noise(sigmoid(V), Striatum_N); I""")
Striatum_ass = zeros((n,n), """dV/dt = (-V + I - Striatum_h)/(Striatum_tau);
                           U = noise(sigmoid(V), Striatum_N); I""")
STN_cog      = zeros((n,1), """dV/dt = (-V + I - STN_h)/(STN_tau);
                           U = noise(V,STN_N); I""")
STN_mot      = zeros((1,n), """dV/dt = (-V + I - STN_h)/(STN_tau);
                           U = noise(V,STN_N); I""")
GPi_cog      = zeros((n,1), """dV/dt = (-V + I - GPi_h)/(GPi_tau);
                           U = noise(V,GPi_N); I""")
GPi_mot      = zeros((1,n), """dV/dt = (-V + I - GPi_h)/(GPi_tau);
                           U = noise(V,GPi_N); I""")
Thalamus_cog = zeros((n,1), """dV/dt = (-V + I - Thalamus_h)/(Thalamus_tau);
                           U = noise(V,Thalamus_N); I""")
Thalamus_mot = zeros((1,n), """dV/dt = (-V + I - Thalamus_h)/(Thalamus_tau);
                           U = noise(V, Thalamus_N); I""")


# Connectivity
# -----------------------------------------------------------------------------
L = DenseConnection( Cortex_cog('U'),   Striatum_cog('I'), 1.0)
init_weights(L)

# Simulate basal learning
L.weights[0] *= 1.050
L.weights[1] *= 1.025

L = DenseConnection( Cortex_mot('U'),   Striatum_mot('I'), 1.0)
init_weights(L)
L = DenseConnection( Cortex_ass('U'),   Striatum_ass('I'), 1.0)
init_weights(L)
L = DenseConnection( Cortex_cog('U'),   Striatum_ass('I'), np.ones((1,2*n+1)))
init_weights(L,0.2)
L = DenseConnection( Cortex_mot('U'),   Striatum_ass('I'), np.ones((2*n+1,1)))
init_weights(L,0.2)
DenseConnection( Cortex_cog('U'),   STN_cog('I'),       1.0 )
DenseConnection( Cortex_mot('U'),   STN_mot('I'),       1.0 )
DenseConnection( Striatum_cog('U'), GPi_cog('I'),      -2.0 )
DenseConnection( Striatum_mot('U'), GPi_mot('I'),      -2.0 )
DenseConnection( Striatum_ass('U'), GPi_cog('I'),      -2.0*np.ones((1,2*n+1)))
DenseConnection( Striatum_ass('U'), GPi_mot('I'),      -2.0*np.ones((2*n+1,1)))
DenseConnection( STN_cog('U'),      GPi_cog('I'),       1.0*np.ones((2*n+1,1)) )
DenseConnection( STN_mot('U'),      GPi_mot('I'),       1.0*np.ones((1,2*n+1)) )
DenseConnection( Cortex_cog('U'),   Thalamus_cog('I'),  0.1 )
DenseConnection( Cortex_mot('U'),   Thalamus_mot('I'),  0.1)

# Faster RT with GPi
# DenseConnection( Thalamus_cog('U'), Cortex_cog('I'),    0.4)
# DenseConnection( Thalamus_mot('U'), Cortex_mot('I'),    0.4)

# Slower RT with GPi
DenseConnection( Thalamus_cog('U'), Cortex_cog('I'),    0.3)
DenseConnection( Thalamus_mot('U'), Cortex_mot('I'),    0.3)

if gpi:
    DenseConnection( GPi_cog('U'),      Thalamus_cog('I'), -0.5 )
    DenseConnection( GPi_mot('U'),      Thalamus_mot('I'), -0.5 )

if cortical:
    K_cog_cog = -0.5 * np.ones((2*n+1,1))
    K_cog_cog[n,0] = 0.5
    K_mot_mot = -0.5 * np.ones((1,2*n+1))
    K_mot_mot[0,n] = 0.5
    K_ass_ass = -0.5 * np.ones((2*n+1,2*n+1))
    K_ass_ass[0,n] = 0.5
    K_cog_ass = 0.20
    K_mot_ass = 0.01
    K_ass_cog = 0.10 * np.ones((1,2*n + 1))
    K_ass_mot = 0.1 * np.ones((2*n + 1, 1))

    DenseConnection( Cortex_cog('U'), Cortex_cog('I'), K_cog_cog)
    DenseConnection( Cortex_mot('U'), Cortex_mot('I'), K_mot_mot)
    DenseConnection( Cortex_ass('U'), Cortex_ass('I'), K_ass_ass)
    C = DenseConnection( Cortex_cog('U'), Cortex_ass('I'), K_cog_ass)
    DenseConnection( Cortex_mot('U'), Cortex_ass('I'), K_mot_ass)
    DenseConnection( Cortex_ass('U'), Cortex_mot('I'), K_ass_mot)
    DenseConnection( Cortex_ass('U'), Cortex_cog('I'), K_ass_cog)

    # Simulate cortical learning
    C.weights[0] *= 1.050
    C.weights[1] *= 1.025


# Trial setup
@clock.at(500*millisecond)
def set_trial(t):
    if familiar:
        m1,m2 = 0,1
    else:
        m1,m2 = 2,3
    while m2 == m1:
        m2 = np.random.randint(4)
    if familiar:
        c1,c2 = 0,1
    else:
        c1,c2 = 2,3
    while c2 == c1:
        c2 = np.random.randint(4)
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0

    v = 7
    Cortex_mot['Iext'][0,m1]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_mot['Iext'][0,m2]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_cog['Iext'][c1,0]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_cog['Iext'][c2,0]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_ass['Iext'][c1,m1] = v + np.random.normal(0,v*Cortex_N)
    Cortex_ass['Iext'][c2,m2] = v + np.random.normal(0,v*Cortex_N)


@clock.at(2500*millisecond)
def reset_trial(t):
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0


# Recording setup
# -----------------------------------------------------------------------------
size        = int(duration/dt) + 1
timesteps   = np.zeros(size)
motor       = np.zeros((5, n, size))
cognitive   = np.zeros((5, n, size))
associative = np.zeros((2, n*n, size))
index       = 0
decision_time = 0

@after(clock.tick)
def register(t):
    global index, decision_time

    # timesteps[index] = t

    if abs(Cortex_mot['U'].max() - Cortex_mot['U'].min()) > 40.0:
        decision_time = t - 500*millisecond
        end()

    # motor[0,:,index] = Cortex_mot['U'].ravel()
    # motor[1,:,index] = Striatum_mot['U'].ravel()
    # motor[2,:,index] = STN_mot['U'].ravel()
    # motor[3,:,index] = GPi_mot['U'].ravel()
    # motor[4,:,index] = Thalamus_mot['U'].ravel()
    # cognitive[0,:,index] = Cortex_cog['U'].ravel()
    # cognitive[1,:,index] = Striatum_cog['U'].ravel()
    # cognitive[2,:,index] = STN_cog['U'].ravel()
    # cognitive[3,:,index] = GPi_cog['U'].ravel()
    # cognitive[4,:,index] = Thalamus_cog['U'].ravel()
    # associative[0,:,index] = Cortex_ass['U'].ravel()
    # associative[1,:,index] = Striatum_ass['U'].ravel()
    # index = index + 1


# Run simulation
D = np.zeros(120)
for i in range(len(D)):
    reset()
    run(time=duration, dt=dt)
    D[i] = decision_time
    print "Trial %d: %.3f" % (i, decision_time)
print "Mean RT: %.3f +/- %.3f" % (D.mean(), D.std())



# Display results
# -----------------------------------------------------------------------------
if 0 and display:
    fig = plt.figure(figsize=(12,5))
    plt.subplots_adjust(bottom=0.15)

    fig.patch.set_facecolor('.9')
    ax = plt.subplot(1,1,1)

    plt.plot(timesteps, cognitive[0,0],c='r', label="Cognitive Cortex")
    plt.plot(timesteps, cognitive[0,1],c='r')
    plt.plot(timesteps, cognitive[0,2],c='r')
    plt.plot(timesteps, cognitive[0,3],c='r')

    plt.plot(timesteps, motor[0,0],c='b', label="Motor Cortex")
    plt.plot(timesteps, motor[0,1],c='b')
    plt.plot(timesteps, motor[0,2],c='b')
    plt.plot(timesteps, motor[0,3],c='b')

    plt.xlabel("Time (seconds)")
    plt.ylabel("Activity (Hz)")
    plt.legend(frameon=False, loc='upper left')
    plt.xlim(0.0,duration)
    #plt.ylim(-5.0,60.0)

    if gpi:
        plt.title("Single trial with GPI")
    else:
        plt.title("Single trial without GPI")

    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
               ['0.0','0.5\n(Trial start)','1.0','1.5', '2.0','2.5\n(Trial stop)','3.0'])
    # plt.savefig("model-results.png")
    plt.show()
