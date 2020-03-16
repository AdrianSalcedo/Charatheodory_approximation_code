import numpy as np
import matplotlib.pyplot as plt
num_sims = 1  # Display five runs
########################################################################################################################
def f_function(y, t):
    deterministic_part = 0
    return deterministic_part

def g_function(y, t):
    stochastic_part = (abs(y)) ** alpha
    return stochastic_part

def dB(delta_t):
    """Sample a random number at each call."""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))
#######################################################################################################################

t_init = 0
t_end  = 1
N      = 1000  # Compute 1000 grid points
dt     = float(t_end - t_init) / N
y_init = 3


alpha = 0.9

ts    = np.arange(t_init, t_end, dt)
ys    = np.zeros(N)

ys[0] = y_init

for _ in range(num_sims):
    for i in range(1, ts.size):
        t = (i-1) * dt
        y = ys[i-1]
        ys[i] = y + f_function(y, t) * dt + g_function(y, t) * dB(dt)
    plt.plot(ts, ys)

########################################################################################################################
n = 5000
dt2  = 1 / n
time = np.linspace(t_init,t_end,n)
xs = np.zeros(n)
xs[0] = y_init
time_0= [t_init-1,time]

for i in range(1,time.size):
    x = xs[i - 1]
    xs[i] = x + f_function(x, t) * dt2 + g_function(x, t) * dB(dt2)

plt.step(time,xs)

plt.show()