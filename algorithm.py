import numpy as np

import numpy.random as npr
import random
from numpy.linalg import inv
from helper_function import *
import ray

def debias(M, tau, Z, l):
    u, s, vh = np.linalg.svd(M, full_matrices = False)
    r = np.sum(s >= 1e-6)
    u = u[:, :r]
    vh = vh[:r, :]
    t1 = l * np.sum(Z*(u.dot(vh))) # lambda * <Z, UV^T>
    PTperpZ = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))
    t2 = np.sum(PTperpZ**2) # ||PTperpZ||_F^2
    M_debias = M + l * u.dot(vh) + t1 / t2 * (Z - PTperpZ) # original M_debias
    return tau - t1 / t2, M_debias, t1, t2

def DC_PR_with_l(O, Z, l, suggest_tau = 0, eps = 1e-6):
    tau = suggest_tau
    num_treat = np.sum(Z)
    for T in range(2000):
        ## update M
        u,s,vh = np.linalg.svd(O - tau*Z, full_matrices=False)
        s = np.maximum(s-l, 0) # possibly related to the analytical form of M hat by (3b) - we may need MH_hat or analytical form not right. 
        M = (u*s).dot(vh)

        tau_new = np.sum(Z * (O - M)) / num_treat # update tau
        # if T % 100 == 0:
        #     print("tau in {} iteration: {}".format(T,tau_new))
        if (np.abs(tau_new - tau) < eps * np.abs(tau)):
            return M, tau, 'successful'
        tau = tau_new
    return M, tau, 'fail'

#gradually decrease l until the rank of the output estimator is more than r
def DC_PR_with_suggested_rank(O, Z, suggest_r = 1):
    coef = 1.1
    pre_tau = np.sum(O*Z)/np.sum(Z)
    ## determine l
    
    _, s, _ = np.linalg.svd(O-pre_tau*Z, full_matrices = False)
    l = s[1]*coef
    #l = 1*coef # change the choice for l
    
    ##inital pre_M and pre_tau for current l
    pre_M, pre_tau, info = DC_PR_with_l(O, Z, l, suggest_tau = pre_tau)
    l = l / coef
    while (True):
        M, tau, info = DC_PR_with_l(O, Z, l, suggest_tau = pre_tau)
        if (info != 'fail' and np.linalg.matrix_rank(M) > suggest_r):
            ###pre_M, pre_tau is the right choice
            tau_debias, M_debias, t1, t2 = debias(pre_M, pre_tau, Z, l*coef)
            return M_debias, tau_debias, pre_M, pre_tau, t1, t2, l
        pre_M = M
        pre_tau = tau
        l = l / coef

def DC_PR_auto_rank(O, Z):
    s = np.linalg.svd(O, full_matrices = False, compute_uv=False)
    suggest_r = np.sum(np.cumsum(s**2) / np.sum(s**2) <= 0.998)
    return DC_PR_with_suggested_rank(O, Z, suggest_r = suggest_r)

def generate_dat(seed, d1, d2, r, tau, noise_sd):
    np.random.seed(seed)
    shape = (d1, d2)
    bl = npr.randn(shape[0], r)
    br = npr.randn(shape[1], r)
    b = bl @ br.T

    z_vec = np.random.normal(0, 1, d1 * d2)
    # replace with a low-rank instrument matrix - assuming d1=d2
    #zz_vec = np.random.normal(0, 4, [d1, r])
    #z_vec = (zz_vec @ zz_vec.T - np.mean(zz_vec @ zz_vec.T)).reshape(-1,)
    e_vec = np.random.normal(0, 1, d1 * d2)

    # treat_vec = (z_vec + e_vec > 1.0)*1.0 # uncomment this for instrument
    treat_vec = (z_vec > 1.0)*1.0 # uncomment this for no-instrument
    e_vec_treat = e_vec[np.where(treat_vec > 0)]
    e_vec_control = e_vec[np.where(treat_vec == 0)]
    obs_treat = create_obs_by_group(treat_vec, d1, 1, b, noise_sd, tau, e_vec_treat)
    obs_control = create_obs_by_group(treat_vec, d1, 0, b, noise_sd, tau, e_vec_control)
    outcome_mat = np.zeros((d1, d2))
    for i in range(len(obs_treat)):
        outcome_mat[obs_treat[i][0], obs_treat[i][1]] = obs_treat[i][2]

    for i in range(len(obs_control)):
        outcome_mat[obs_control[i][0], obs_control[i][1]] = obs_control[i][2]

    treat_mat = np.reshape(treat_vec, [d1, d2])
    return b, z_vec, treat_mat, outcome_mat

@ray.remote
def worker(seed, d1, d2, r, tau, noise_sd):
    b, z_vec, treat_mat, outcome_mat = generate_dat(seed, d1, d2, r, tau, noise_sd)
    O = outcome_mat
    Z = treat_mat
    M, tau, M_raw, tau_raw, t1, t2, l = DC_PR_auto_rank(O, Z)
    return M, tau, M_raw, tau_raw, b, t1, t2, l

def opt_instrument(z_vec, treat_mat):
    d1, d2 = treat_mat.shape
    # now we correct for the outcome matrix and treatment matrix based on P_Z
    error = []
    for i in np.arange(0, 3, 0.05):
        h_z_vec = (np.reshape(z_vec, [len(z_vec), 1]) > i)*1.0

        P_Z = h_z_vec.dot(inv(h_z_vec.T.dot(h_z_vec)).dot(h_z_vec.T))

        # outcome_mat_corrected = np.reshape(P_Z.dot(outcome_mat.reshape([d1*d2, 1])), [d1, d2])
        treat_mat_corrected = np.reshape(P_Z.dot(treat_mat.reshape([d1*d2, 1])), [d1, d2])
        error.append(np.sum((treat_mat !=  treat_mat_corrected)))
    placeholder = np.arange(0, 3, 0.05)[np.argmin(error)-10]
        
    h_z_vec = (np.reshape(z_vec, [len(z_vec), 1]) > placeholder)*1.0
    return h_z_vec

@ray.remote
def worker_proj(seed, d1, d2, r, tau, noise_sd):
    b, z_vec, treat_mat, outcome_mat = generate_dat(seed, d1, d2, r, tau, noise_sd)
    h_z_vec = opt_instrument(z_vec, treat_mat)
    P_Z = h_z_vec.dot(inv(h_z_vec.T.dot(h_z_vec)).dot(h_z_vec.T))
    O_corrected = np.reshape(P_Z.dot(outcome_mat.reshape([d1*d2, 1])), [d1, d2])
    Z_corrected = np.reshape(P_Z.dot(treat_mat.reshape([d1*d2, 1])), [d1, d2])
    # M, tau, M_raw, tau_raw, t1, t2, l = DC_PR_auto_rank(O_corrected, Z_corrected)
    # using a suggested rank of r since we know the rank
    M, tau, M_raw, tau_raw, t1, t2, l = DC_PR_with_suggested_rank(O_corrected, Z_corrected, suggest_r = r)
    return M, tau, M_raw, tau_raw, b, t1, t2, l

@ray.remote
def worker_cmm(seed, d1, d2, r, tau, noise_sd):
    b, z_vec, treat_mat, outcome_mat = generate_dat(seed, d1, d2, r, tau, noise_sd)
    h_z_vec = opt_instrument(z_vec, treat_mat)
    # P_Z = h_z_vec.dot(inv(h_z_vec.T.dot(h_z_vec)).dot(h_z_vec.T))
    h_z_mat = h_z_vec.reshape([d1, d2])
    O_corrected = h_z_mat * outcome_mat
    T_corrected = h_z_mat * treat_mat
    # M, tau, M_raw, tau_raw, t1, t2, l = DC_PR_auto_rank(O_corrected, Z_corrected)
    # using a suggested rank of r since we know the rank
    M, tau, M_raw, tau_raw, t1, t2, l = DC_PR_with_suggested_rank(O_corrected, T_corrected, suggest_r = r)
    return M, tau, M_raw, tau_raw, b, t1, t2, l

@ray.remote
def worker_2sls(seed, d1, d2, r, tau, noise_sd):
    b, z_vec, treat_mat, outcome_mat = generate_dat(seed, d1, d2, r, tau, noise_sd)
    h_z_vec = opt_instrument(z_vec, treat_mat)
    P_Z = h_z_vec.dot(inv(h_z_vec.T.dot(h_z_vec)).dot(h_z_vec.T))
    treat_vec = treat_mat.reshape([d1*d2, 1])
    outcome_vec = outcome_mat.reshape([d1*d2, 1])
    tau = inv(treat_vec.T @ P_Z @ treat_vec) * (treat_vec.T @ P_Z @ outcome_vec)
    return tau

@ray.remote
def worker_ols(seed, d1, d2, r, tau, noise_sd):
    b, z_vec, treat_mat, outcome_mat = generate_dat(seed, d1, d2, r, tau, noise_sd)
    treat_vec = treat_mat.reshape([d1*d2, 1])
    outcome_vec = outcome_mat.reshape([d1*d2, 1])
    tau = inv(treat_vec.T @ treat_vec) * (treat_vec.T @ outcome_vec)
    return tau