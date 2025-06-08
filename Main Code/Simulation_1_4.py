import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fsolve
from sklearn.metrics import mean_absolute_error
from Multiply_updates import MultiplicativeUpdates

# Traditional Project Gradient Descent
def pgd_rectangle(low_bounds, upon_bounds, A, b, run_am, linear_rate = 0.01):
    x = np.mean([low_bounds, upon_bounds],axis=0)
    for k in range(run_am):
        grad = A @ x+ b
        x = x - linear_rate * grad
        x = np.minimum(np.maximum(x, low_bounds), upon_bounds)
    return x

def disc_project(x, center, r):
    dist_x_cent = x - center
    norm_sq = np.dot(dist_x_cent, dist_x_cent)
    if norm_sq <= r:
        return x
    else:
        return center + dist_x_cent * np.sqrt(r) / np.sqrt(norm_sq)

def pgd_disc(center, r, A, b, run_am, linear_rate = 0.01, lambda_pgd=0, lambda_linear_rate=0.1):
    x = center
    for k in range(run_am):
        grad = A @ x + b + 2*lambda_pgd*(x-center)
        x = x - linear_rate * grad
        x = disc_project(x, center, r)
        lambda_pgd = lambda_pgd + lambda_linear_rate * (np.dot(x-center,x-center) - r)
        lambda_pgd = np.max(lambda_pgd, 0)
    return x

def ellipse_project(x, center, eta, r):
    eta = eta.reshape(-1,)
    x = x.reshape(-1,)
    center = center.reshape(-1,)
    dist_x_cent = (x - center)*eta
    norm_sq = np.dot(dist_x_cent, dist_x_cent)
    if norm_sq <= r**2:
        return x
    else:
        return center + r * dist_x_cent /eta / np.sqrt(norm_sq)

def pgd_ellipse(center, r, A, b, eta, run_am, linear_rate = 0.01, lambda_pgd=0, lambda_linear_rate=0.1):
    x = center
    for k in range(run_am):
        grad = A @ x + b + 2*lambda_pgd*(x-center)*eta*eta
        x = x - linear_rate * grad
        x = ellipse_project(x, center, eta, r)
        lambda_pgd = lambda_pgd + lambda_linear_rate * (np.dot(eta*(x-center),eta*(x-center)) - r**2)
        lambda_pgd = np.max(lambda_pgd, 0)
    return x

def simplex_project(x, z):
    x = np.maximum(x, 0)
    if np.sum(x) <= z:
        return x
    else:
        # Compute theta by Michelot algorithm
        u = np.sort(x)[::-1]
        u_cum = np.cumsum(u)
        rho = np.where(u > (u_cum - z) / (np.arange(len(u)) + 1))[0][-1]
        theta = (u_cum[rho] - z) / (rho + 1)
        return np.maximum(x - theta, 0)
    
def pgd_linear(c, A, b, run_am, linear_rate = 0.01, lambda_pgd=0, lambda_linear_rate=0.1):
    x = np.ones(shape=(A.shape[0],))*c/A.shape[0]*0.5
    for k in range(run_am):
        grad = A @ x + b + lambda_pgd
        x = x - linear_rate * grad
        x = simplex_project(x, c)
        lambda_pgd = lambda_pgd + lambda_linear_rate * (np.sum(x) - c)
        lambda_pgd = np.max(lambda_pgd, 0)
    return x

def run_main(ds, run_am=1000):
    new_m_ex1, trad_m_ex1, target_ex1 = [], [], []
    new_m_ex2, trad_m_ex2, target_ex2 = [], [], []
    new_m_ex3, trad_m_ex3, target_ex3 = [], [], []
    new_m_ex4, trad_m_ex4, target_ex4 = [], [], []
    for i in tqdm(range(500)):
        # Fix the random, seed from 0 to 500
        np.random.seed(i)
        # Ex1
        low_bounds = np.random.randint(2, 10, size=(ds,))
        upon_bounds = low_bounds + np.random.randint(1,10, size=(ds,))
        x_target = np.random.uniform(low_bounds, upon_bounds, size=(low_bounds.shape[0]))
        Q, _ = np.linalg.qr(np.random.randn(ds, ds))
        A = Q.dot(np.diag(np.abs(np.random.randn(ds,)))).dot(np.transpose(Q))
        b = -A.dot(x_target)
        
        MU_th = MultiplicativeUpdates(A, b, domain='rectangle', L_bounds=low_bounds, U_bounds=upon_bounds, run_am=run_am)
        MU_out = MU_th.MU_main()
        PGD_out = pgd_rectangle(low_bounds, upon_bounds, A, b, run_am, linear_rate=0.01)
        new_m_ex1.append(MU_out.reshape(-1,))
        target_ex1.append(x_target.reshape(-1,))
        trad_m_ex1.append(PGD_out)

        # Ex2
        r = 3
        center = np.random.uniform(0,5,size=(ds,)) + r*1.5
        x_target = center + np.random.uniform(-5/ds,5/ds,size=(ds,))
        while LA.norm(x_target - center)**2 > r**2:
            x_target = center + np.random.uniform(-2/ds,2/ds,size=(ds,))
        Q, _ = np.linalg.qr(np.random.randn(ds, ds))
        A = Q.dot(np.diag(np.abs(np.random.randn(ds,)))).dot(np.transpose(Q))
        b = -A.dot(x_target)

        MU_th = MultiplicativeUpdates(A, b, domain='disc', r=r, center=center, run_am=run_am)
        MU_out = MU_th.MU_main()
        PGD_out = pgd_disc(center, r, A, b, run_am, linear_rate=0.01)
        new_m_ex2.append(MU_out.reshape(-1,))
        target_ex2.append(x_target.reshape(-1,))
        trad_m_ex2.append(PGD_out)

        # Ex3
        r = 3
        center = np.random.uniform(0,5,size=(ds,)) + r*1.5
        eta = np.array([3,4,5,4,3,2,1,1,2])[:ds]
        x_target = center + np.random.uniform(-5/ds,5/ds,size=(ds,))
        while LA.norm(eta*(x_target - center))**2 > r**2:
            x_target = center + np.random.uniform(-2/ds,2/ds,size=(ds,))
        Q, _ = np.linalg.qr(np.random.randn(ds, ds))
        A = Q.dot(np.diag(np.abs(np.random.randn(ds,)))).dot(np.transpose(Q))
        b = -A.dot(x_target)

        MU_th = MultiplicativeUpdates(A, b, domain='ellipse', r=r, center=center, eta=eta, run_am=run_am)
        MU_out = MU_th.MU_main()
        PGD_out = pgd_ellipse(center, r, A, b, eta, run_am, linear_rate=0.01)
        new_m_ex3.append(MU_out.reshape(-1,))
        target_ex3.append(x_target.reshape(-1,))
        trad_m_ex3.append(PGD_out)

        # Ex4
        c = 5
        x_target = np.ones(shape=(ds,))*c/ds/2 + np.random.uniform(0,1,size=(ds,))
        j = 1
        while np.sum(x_target)>c:
            x_target = np.ones(shape=(ds,))*c/ds/2 + np.random.uniform(0,1/j,size=(ds,))
            j = j+0.1
        Q, _ = np.linalg.qr(np.random.randn(ds, ds))
        A = Q.dot(np.diag(np.abs(np.random.randn(ds,)))).dot(np.transpose(Q))
        b = -A.dot(x_target)
        
        MU_th = MultiplicativeUpdates(A, b, domain='linear', linear_C=c, run_am=run_am)
        MU_out = MU_th.MU_main()
        PGD_out = pgd_linear(c, A, b, run_am, linear_rate=0.01)
        new_m_ex4.append(MU_out.reshape(-1,))
        target_ex4.append(x_target.reshape(-1,))
        trad_m_ex4.append(PGD_out)

    return [np.array(target_ex1), np.array(new_m_ex1), np.array(trad_m_ex1)],\
           [np.array(target_ex2), np.array(new_m_ex2), np.array(trad_m_ex2)],\
           [np.array(target_ex3), np.array(new_m_ex3), np.array(trad_m_ex3)],\
           [np.array(target_ex4), np.array(new_m_ex4), np.array(trad_m_ex4)]

# Visualization process
def exa_process(ex_res):
    tar, new_m, trad_m = ex_res
    return {'MU': np.around(mean_absolute_error(np.array(tar), np.array(new_m)), 7),
            'PGDL': np.around(mean_absolute_error(np.array(tar), np.array(trad_m)),7)}

def dict_process(*dicts):
    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key in merged_dict:
                if not isinstance(merged_dict[key], list):
                    merged_dict[key] = [merged_dict[key]]
                merged_dict[key].append(value)
            else:
                merged_dict[key] = value
    return merged_dict

ex1_res_ds3, ex2_res_ds3, ex3_res_ds3, ex4_res_ds3 = run_main(3, run_am=800)
ex1_res_ds5, ex2_res_ds5, ex3_res_ds5, ex4_res_ds5 = run_main(5, run_am=800)
ex1_res_ds7, ex2_res_ds7, ex3_res_ds7, ex4_res_ds7 = run_main(7, run_am=800)
ex1_res_ds9, ex2_res_ds9, ex3_res_ds9, ex4_res_ds9 = run_main(9, run_am=800)

MAE_ex1_ds3 = exa_process(ex1_res_ds3)
MAE_ex2_ds3 = exa_process(ex2_res_ds3)
MAE_ex3_ds3 = exa_process(ex3_res_ds3)
MAE_ex4_ds3 = exa_process(ex4_res_ds3)

MAE_ex1_ds5 = exa_process(ex1_res_ds5)
MAE_ex2_ds5 = exa_process(ex2_res_ds5)
MAE_ex3_ds5 = exa_process(ex3_res_ds5)
MAE_ex4_ds5 = exa_process(ex4_res_ds5)

MAE_ex1_ds7 = exa_process(ex1_res_ds7)
MAE_ex2_ds7 = exa_process(ex2_res_ds7)
MAE_ex3_ds7 = exa_process(ex3_res_ds7)
MAE_ex4_ds7 = exa_process(ex4_res_ds7)

MAE_ex1_ds9 = exa_process(ex1_res_ds9)
MAE_ex2_ds9 = exa_process(ex2_res_ds9)
MAE_ex3_ds9 = exa_process(ex3_res_ds9)
MAE_ex4_ds9 = exa_process(ex4_res_ds9)

dict_ds3 = dict_process(MAE_ex1_ds3, MAE_ex2_ds3, MAE_ex3_ds3, MAE_ex4_ds3)
dict_ds5 = dict_process(MAE_ex1_ds5, MAE_ex2_ds5, MAE_ex3_ds5, MAE_ex4_ds5)
dict_ds7 = dict_process(MAE_ex1_ds7, MAE_ex2_ds7, MAE_ex3_ds7, MAE_ex4_ds7)
dict_ds9 = dict_process(MAE_ex1_ds9, MAE_ex2_ds9, MAE_ex3_ds9, MAE_ex4_ds9)

x_ticks = ("Example 1", "Example 2", "Example 3", "Example 4")
width = 0.25

fig, ax = plt.subplots(layout='constrained', nrows=2, ncols=2, figsize=(10, 6))
multiplier = 0
for attribute, measurement in dict_ds3.items():
    offset = width * multiplier
    rects = ax[0,0].bar(np.arange(len(x_ticks))  + offset, measurement, width, label=attribute)
    multiplier += 1
ax[0,0].set_ylabel('MAE of predict and target')
ax[0,0].set_title('Target variable dimension: {ds}'.format(ds=3))
ax[0,0].set_xticks([])
ax[0,0].set_yscale('log')
ax[0,0].grid()

multiplier = 0
for attribute, measurement in dict_ds5.items():
    offset = width * multiplier
    rects = ax[0,1].bar(np.arange(len(x_ticks))  + offset, measurement, width)
    multiplier += 1
ax[0,1].set_title('Target variable dimension: {ds}'.format(ds=5))
ax[0,1].set_xticks([])
ax[0,1].set_yscale('log')
ax[0,1].grid()

multiplier = 0
for attribute, measurement in dict_ds7.items():
    offset = width * multiplier
    rects = ax[1,0].bar(np.arange(len(x_ticks))  + offset, measurement, width)
    multiplier += 1
ax[1,0].set_ylabel('MAE of predict and target')
ax[1,0].set_title('Target variable dimension: {ds}'.format(ds=7))
ax[1,0].set_yscale('log')
ax[1,0].set_xticks(np.arange(len(x_ticks)) + width, x_ticks)
ax[1,0].grid()

multiplier = 0
for attribute, measurement in dict_ds9.items():
    offset = width * multiplier
    rects = ax[1,1].bar(np.arange(len(x_ticks))  + offset, measurement, width)
    multiplier += 1
ax[1,1].set_title('Target variable dimension: {ds}'.format(ds=9))
ax[1,1].set_yscale('log')
ax[1,1].set_xticks(np.arange(len(x_ticks)) + width, x_ticks)
ax[1,1].grid()

fig.legend(loc='outside upper center', ncols=2, bbox_to_anchor=(0.5, 1.05))
fig.suptitle('MAE of Multiplicative Update vs Project Gradient Descent with Lagrangian', fontsize=16, fontweight="bold", y=1.08)
plt.show()