import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def rhs(y, t_zero):
    s_j_p = y[0]
    s_a_p = y[1]
    l_j_p = y[2]
    l_a_p = y[3]
    i_j_p = y[4]
    i_a_p = y[5]
    s_v = y[6]
    i_v = y[7]

    s_j_p_prime = - beta_j_p * s_j_p * i_v + r_1_j * l_j_p + r_2_j * i_j_p+ r_a * i_a_p - alpha * s_j_p
    s_a_p_prime = - beta_a_p * s_a_p * i_v + alpha * s_j_p
    l_j_p_prime = beta_j_p * s_j_p * i_v - b_j * l_j_p - r_1_j * l_j_p
    l_a_p_prime = beta_a_p * s_a_p * i_v - b_a * l_a_p
    i_j_p_prime = b_j * l_j_p - r_2_j * i_j_p
    i_a_p_prime = b_a * l_a_p - r_a * i_a_p
    s_v_prime = - beta_j_v * s_v * i_j_p - beta_a_v * s_v * i_a_p - gamma * s_v + (1-theta) *  mu
    i_v_prime = beta_j_v * s_v * i_j_p + beta_a_v * s_v * i_a_p - gamma * i_v + theta * mu
    rhs_np_array = np.array([s_j_p_prime, s_a_p_prime, l_j_p_prime, l_a_p_prime, i_j_p_prime, i_a_p_prime, s_v_prime, i_v_prime])
    return (rhs_np_array)

beta_j_p = 0.5
r_1_j = 0.0035
r_2_j = 0.0035
r_a = 0.003
beta_a_p = 0.5
b_j = 0.025
b_a = 0.050
beta_j_v = 0.00015
beta_a_v = 0.00015
gamma = 0.06
theta = 0.2
mu = 0.3
alpha = 0.5

y_zero = np.array([0.7, 0.3, 0.0, 0.0, 0.0 , 0.0, 0.92, 0.08])
t = np.linspace(0, 120, 1000)
sol = odeint(rhs, y_zero, t)
plt.plot(t, sol[:, 4], 'b', label="$I_j_p$")
plt.plot(t, sol[:, 5], 'g', label="$I_a_p$")
#plt.plot([0.6+7, 1.1+13, 1.7+21, 2.4+28,2.8+35,3.4+42], [0.005, 0.007, 0.008, 0.02,.17+0.4,0.031+0.8], 'ro')
#plt.plot(t, sol[:, 4], 'r', label='$I_v$')
plt.tight_layout()
plt.xlabel('$t$')
plt.ylabel('proporción de plantas infectadas')
plt.ylim(-0.05,1)
plt.xlim(0,70)
plt.grid()
plt.show()
