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

def brownian_path_sampler(step_size,number_max_of_steps):
    normal_sampler = np.sqrt(step_size)*np.random.randn(number_max_of_steps)
    w_t = np.zeros(number_max_of_steps+1)
    w_t[1:] = np.cumsum(normal_sampler)

    return (normal_sampler,w_t)
#######################################################################################################################

t_init = 0
t_end  = 1
N      = 100000  # Compute 1000 grid points
dt     = float(t_end - t_init) / N
y_init = 3
alpha = 0.9
########################################################################################################################
dB,B_t = brownian_path_sampler(dt,N)


ts    = np.arange(t_init, t_end, dt)
ys    = np.zeros(N)

ys[0] = y_init

for _ in range(num_sims):
    for i in range(1, ts.size):
        t = (i-1) * dt
        y = ys[i-1]
        ys[i] = y + f_function(y, t) * dt + g_function(y, t) * dB[i-1]
    plt.plot(ts, ys)

########################################################################################################################
p = 2**10 # scale for new step size
Dt = p*dt # new step size
L = int(N/p) # new number of step size

new_time = np.linspace(t_init,t_end,L) # new discretization of time by L

B_t_aux = 0
Binc = []
Binc.append(B_t_aux)


for j in np.arange(L): #Here we do the increment in the new step size
    B_t_aux = np.sum(dB[j*p:(j+1)*p]) # operation for generate the increment
    Binc.append(B_t_aux)  # Vector with the increments

Binc = np.array(Binc) # Transform the vector list to vector numeric
Binc_1 = Binc.cumsum() # do the cumulative sum in the new step size

xs = np.zeros(L)
xs[0] = y_init


for i in range(1,new_time.size):
    x = xs[i - 1]
    xs[i] = x + f_function(x, t) * Dt + g_function(x, t) * Binc[i-1]

plt.step(new_time,xs)

plt.show()