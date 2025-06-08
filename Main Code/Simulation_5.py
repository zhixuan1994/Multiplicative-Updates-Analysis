import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fsolve

def M_matrix(A, b, v):
    A_p = np.where(A >= 0, A, 0)
    A_n = np.abs(np.where(A < 0, A, 0))
    a_vec = A_p.dot(v)
    c_vec = A_n.dot(v)
    a_vec = a_vec.reshape(-1,)
    for i in range(len(a_vec)):
        if np.abs(a_vec[i]) < 1e-10:
            a_vec[i] = 1e-10
    c_vec = c_vec.reshape(-1,)
    b_vec = b.reshape(-1,)
    temp = b_vec**2 + 4*a_vec*c_vec
    m = -b_vec + np.sqrt(np.max(np.concatenate([0.0001*np.ones(shape=(len(temp),1)),temp.reshape(-1,1)], axis=1), axis=1))
    m = m / (2*a_vec)
    return np.diag(m)

def a_star_fun_ex5(a, M, v, b):
    Mv = M.dot(v)
    v_temp = a*v + (1-a)*Mv
    return v_temp[0]**2 + v_temp[1]**2 - 2*v_temp[0]*v_temp[1] - b

# Calculate a^*, initial from 0.5
def a_star_cal_ex5(v, M, b):
    return fsolve(a_star_fun_ex5, 0.5, args=(M, v, b))

def calculate_a_ex5(v, M, b_1, b_2):
    Mv = M.dot(v)
    if (Mv[0] - Mv[1])**2 > b_2:
        return a_star_cal_ex5(v, M, b_2)
    else:
        return a_star_cal_ex5(v, M, b_1)

def v_update_ex5(M, v, b_1, b_2):
    Mv = np.dot(M, v)
    check_temp = (Mv[0] - Mv[1])**2
    if check_temp - b_2 > -1e-3 or np.any(Mv - b_1 < 1e-5):
        a= calculate_a_ex5(v, M, b_1, b_2)
        return (a*v + (1-a)*Mv + v)/2
    else:
        return Mv

def main_ex5(A, b, v, b_1, b_2):
    b = b.reshape(-1,1)
    v = v.reshape(-1,1)
    M = M_matrix(A, b, v)
    return v_update_ex5(M, v, b_1, b_2)

def test_trad_ex5(initial_val, b_1, b_2, A, b, run_am, eta = 0.01, lambda_ex5=0, eta_lambda_ex5=0.1):
    b = b.reshape(-1,)
    x = initial_val
    for k in range(run_am):
        grad = A @ x + b + 2*lambda_ex5* (x[0] - x[1])*np.array([1,-1])
        x = x - eta * grad
        x = np.maximum(x, b_1)
        if (x[0] - x[1])**2 > b_2:
            if x[0] > x[1]:
                x[0] = x[1] + np.sqrt(b_2)
            else:
                x[1] = x[0] + np.sqrt(b_2)
        lambda_ex5 = lambda_ex5 + eta_lambda_ex5 * ((x[0]-x[1])**2 - b_2)
        lambda_ex5 = np.max([lambda_ex5, 0])
    return x

def test_main_ex5(A, b, initial_val, b_1, b_2, run_am, eta = 0.01):
    temp = main_ex5(A, b, initial_val, b_1, b_2)
    new_method = [temp.reshape(-1,)]
    obj = [initial_val.reshape(-1,), temp.reshape(-1,)]
    for i in range(run_am):
        temp = main_ex5(A, b, temp, b_1, b_2)
        new_method.append(temp.reshape(-1,))
        obj.append(temp.reshape(-1,))
    return new_method[-1], test_trad_ex5(initial_val, b_1, b_2, A, b, run_am, eta = 0.01)

ds = 2
b_1, b_2 = 0, 4
run_am = 800
x_target_list = np.array([[4,2.5], [6, 5.5], [8, 8.5], [10,11.5]])
MU_out, PGD_out = [], []

for x_target in tqdm(x_target_list):
    MU_temp, trad_temp = [], []
    for i in range(500):
        np.random.seed(i)
        Q, _ = np.linalg.qr(np.random.randn(ds, ds))
        A = Q.dot(np.diag(np.abs(np.random.randn(ds,)))).dot(np.transpose(Q))
        b = -A.dot(x_target)
        initial_val = np.ones(shape=(ds,))
        New_m, Trad_m = test_main_ex5(A, b, initial_val, b_1, b_2, run_am)
        MU_temp.append(New_m)
        trad_temp.append(Trad_m)
    MU_out.append(MU_temp)
    PGD_out.append(trad_temp)
MU_out = np.array(MU_out)
PGD_out = np.array(PGD_out)

color_li = ['.b', '.r', '.y', '.c']
label_li = ['Case 1', 'Case 2', 'Case 3', 'Case 4']
fig, ax = plt.subplots(layout='constrained', nrows=2, ncols=2, figsize=(10, 6))

fig.suptitle('v estimation of MU (left) vs PGDL (right) on Example 5', fontsize=16, fontweight="bold", y=1.08)

for i in range(4):
    ax[0,0].plot(MU_out[i][:,0], color_li[i], markersize=2, label=label_li[i])
    ax[0,1].plot(PGD_out[i][:,0], color_li[i], markersize=2)
    ax[1,0].plot(MU_out[i][:,1], color_li[i], markersize=2)
    ax[1,1].plot(PGD_out[i][:,1], color_li[i], markersize=2)

ax[0,0].set_xticks([])
ax[0,0].set_ylabel('Estimation of v')
ax[0,0].grid()
ax[0,1].set_xticks([])
ax[0,1].grid()
ax[1,0].set_ylabel('Estimation of v')
ax[1,0].set_xlabel('Run test')
ax[1,0].grid()
ax[1,1].set_xlabel('Run test')
ax[1,1].grid()
fig.legend(loc='outside upper center', ncols=4, bbox_to_anchor=(0.5, 1.05))