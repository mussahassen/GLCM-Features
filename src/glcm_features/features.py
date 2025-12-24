import numpy as np
from numba import njit
from skimage.feature import graycomatrix, graycoprops


def compute_glcms(gs_image, levels = 2):

    glcm0 = graycomatrix(gs_image, distances=[1], angles=[0], levels=levels, symmetric=False)
    glcm45 = graycomatrix(gs_image, distances=[1], angles=[np.pi/4], levels=levels, symmetric=False)
    glcm90 = graycomatrix(gs_image, distances=[1], angles=[np.pi/2], levels=levels, symmetric=False)
    glcm135 = graycomatrix(gs_image, distances=[1], angles=[3 * np.pi/4], levels=levels, symmetric=False)
    mean_glcm = (glcm0 + glcm45 + glcm90 + glcm135)/4

    glcms = {'mean': mean_glcm, '0': glcm0, '45': glcm45, '90': glcm90, '135': glcm135}

    return glcms


@njit(cache=True)
def pre_feature_statistics(mean_glcm):

    normed_glcm = mean_glcm / np.sum(mean_glcm)
    N = normed_glcm.shape[0]
    i_indices = np.arange(N) + 1
    j_indices = np.arange(N) + 1
    
    # Marginal probabilities
    p_x = np.sum(normed_glcm, axis=1)  # Sum over j
    p_x = p_x.squeeze()
    p_y = np.sum(normed_glcm, axis=0)  # Sum over i
    p_y = p_y.squeeze()
    
    # Mean values
    indices = np.arange(N) + 1
    mu_x = np.dot(indices, p_x)
    mu_y = np.dot(indices, p_y)
    mu = (mu_x + mu_y) / 2
    
    # Standard deviations
    sigma_x = np.sqrt(np.sum(((i_indices - mu_x) ** 2) * p_x))
    sigma_y = np.sqrt(np.sum(((j_indices - mu_y) ** 2) * p_y))
    
    # HX and HY
    p_x_nonzero = p_x[p_x > 0]
    HX = -np.sum(p_x_nonzero * np.log2(p_x_nonzero))
    
    p_y_nonzero = p_y[p_y > 0]
    HY = -np.sum(p_y_nonzero * np.log2(p_y_nonzero))
    
    # p_x+y and p_x-y 
    p_sum = np.zeros(2 * N - 1)
    p_diff = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            # p_x+y
            p_sum[i + j] += normed_glcm[i, j]
            # p_x-y
            p_diff[abs(i - j)] += normed_glcm[i, j]
    
    stats = {
        'Pij' : normed_glcm,
        'N': N,
        'mu_x': mu_x,
        'mu_y': mu_y,
        'mu': mu,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'p_x': p_x,
        'p_y': p_y,
        'HX': HX,
        'HY': HY,
        'p_sum': p_sum,
        'p_diff': p_diff
    }
    
    return stats




## ---------------------------------------------
## Features
## ---------------------------------------------

def compute_autocorrelation(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]

    autocorrelation = 0.0

    for i in range(N):
        for j in range(N):
            i_1 = i + 1
            j_1 = j + 1

            # φ(i,j,g(P)) = i·j
            phi = i_1 * j_1

            # ψ(p(i,j)) = p(i,j)
            psi = normed_glcm[i,j]

            autocorrelation += phi * psi

    return autocorrelation


def compute_cluster_prominence(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    mu =  stats['mu']

    cluster_prominence = 0.0

    for i in range(N):
        for j in range(N):
            i_1 = i + 1
            j_1 = j + 1
            
            # φ(i,j,g(P)) = (i + j - 2*mu)^3 
            phi = (i_1 + j_1 - 2 * mu)**3

            # ψ(p(i,j)) = p(i,j)
            psi = normed_glcm[i,j]
            
            cluster_prominence += phi * psi
            
    return cluster_prominence


def compute_cluster_shade(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    mu =  stats['mu']

    cluster_shade = 0.0

    for i in range(N):
        for j in range(N):
            i_1 = i + 1
            j_1 = j + 1
            
            # φ(i,j,g(P)) = (i + j - 2*mu)^4 
            phi = (i_1 + j_1 - 2 * mu)**4

            # ψ(p(i,j)) = p(i,j)
            psi = normed_glcm[i,j]
            
            cluster_shade += phi * psi
    
    return cluster_shade


def compute_dissimilarity(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    
    dissimilarity = 0.0
    
    for i in range(N):
        for j in range(N):
            i_1 = i + 1
            j_1 = j + 1
            
            # φ(i,j,g(P)) = |i - j|
            phi = abs(i_1 - j_1)
            
            # ψ(p(i,j)) = p(i,j)
            psi = normed_glcm[i, j] 
            
            dissimilarity += phi * psi
    
    return dissimilarity


def compute_entropy(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    
    entropy_val = 0.0
    
    for i in range(N):
        for j in range(N):
            p = normed_glcm[i, j]
            if p > 0:
                # φ(i,j,g(P)) = 1
                phi = 1
                
                # ψ(p(i,j)) = -p(i,j) * log(p(i,j))
                psi = -p * np.log(p)
                
                entropy_val += phi * psi
    
    return entropy_val

def compute_difference_entropy(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    
    # p_x-y(k)
    p_diff = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            k = abs((i + 1) - (j + 1))
            if k < N:
                p_diff[k] += normed_glcm[i, j]
    

    diff_entropy = 0.0
    for k in range(N):
        if p_diff[k] > 0:
            diff_entropy -= p_diff[k] * np.log(p_diff[k])
    
    return diff_entropy


def compute_difference_variance(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    
    # p_x-y(k)
    p_diff = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            k = abs((i + 1) - (j + 1))
            if k < N:
                p_diff[k] += normed_glcm[i, j]
    
    # mean
    k_values = np.arange(N)
    mu_diff = np.sum(k_values * p_diff)
    
    # variance
    diff_variance = 0.0
    for k in range(N):
        diff_variance += ((k - mu_diff) ** 2) * p_diff[k]
    
    return diff_variance


def compute_inverse_difference(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    
    inv_diff = 0.0
    
    for i in range(N):
        for j in range(N):
            i_1based = i + 1
            j_1based = j + 1
            
            # φ(i,j,g(P)) = 1
            phi = 1
            
            # ψ(p(i,j)) = p(i,j) / (1 + |i-j|)
            psi = normed_glcm[i, j] / (1 + abs(i_1based - j_1based))
            
            inv_diff += phi * psi
    
    return inv_diff


def compute_sum_average(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]

    # Compute p_x+y(k) for k = 2 to 2N
    p_sum = np.zeros(2 * N + 1)
    
    for i in range(N):
        for j in range(N):
            k = (i + 1) + (j + 1)
            if k <= 2 * N:
                p_sum[k] += normed_glcm[i, j]
    
    # Compute sum average
    sum_avg = 0.0
    for k in range(2, 2 * N + 1):
        sum_avg += k * p_sum[k]
    
    return sum_avg


def compute_sum_entropy(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    
    p_sum = np.zeros(2 * N + 1)
    
    for i in range(N):
        for j in range(N):
            k = (i + 1) + (j + 1)
            if k <= 2 * N:
                p_sum[k] += normed_glcm[i, j]
    
    # Compute sum entropy
    sum_ent = 0.0
    for k in range(2, 2 * N + 1):
        if p_sum[k] > 0:
            sum_ent -= p_sum[k] * np.log(p_sum[k])
    
    return sum_ent


def compute_sum_of_squares(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    mu_x = stats['mu_x']
    
    sum_squares = 0.0
    
    for i in range(N):
        for j in range(N):
            i_1based = i + 1
            
            # φ(i,j,g(P)) = (i - μ_x)^2
            phi = (i_1based - mu_x) ** 2
            
            # ψ(p(i,j)) = p(i,j)
            psi = normed_glcm[i, j]
            
            sum_squares += phi * psi
    
    return sum_squares


def compute_sum_variance(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    
    # Compute p_x+y(k) for k = 2 to 2N
    p_sum = np.zeros(2 * N + 1)
    
    for i in range(N):
        for j in range(N):
            k = (i + 1) + (j + 1)
            if k <= 2 * N:
                p_sum[k] += normed_glcm[i, j]
    
    # Compute sum average
    sum_avg = 0.0
    for k in range(2, 2 * N + 1):
        sum_avg += k * p_sum[k]
    
    # Compute sum variance
    sum_var = 0.0
    for k in range(2, 2 * N + 1):
        sum_var += ((k - sum_avg) ** 2) * p_sum[k]
    
    return sum_var


def compute_information_measure_correlation_1(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    p_x = stats['p_x']
    p_y = stats['p_y']
    
    # HXY: Entropy of GLCM
    HXY = 0.0
    for i in range(N):
        for j in range(N):
            if normed_glcm[i, j] > 0:
                HXY -= normed_glcm[i, j] * np.log(normed_glcm[i, j])
    
    # HX: Entropy of p_x
    HX = 0.0
    for i in range(N):
        if p_x[i] > 0:
            HX -= p_x[i] * np.log(p_x[i])
    
    # HY: Entropy of p_y
    HY = 0.0
    for j in range(N):
        if p_y[j] > 0:
            HY -= p_y[j] * np.log(p_y[j])
    
    # HXY1: sum over p(i,j) * log(p_x(i) * p_y(j))
    HXY1 = 0.0
    for i in range(N):
        for j in range(N):
            if normed_glcm[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                HXY1 -= normed_glcm[i, j] * np.log(p_x[i] * p_y[j])
    
    # IMC1
    max_HX_HY = max(HX, HY)
    if max_HX_HY > 0:
        imc1 = (HXY - HXY1) / max_HX_HY
    else:
        imc1 = 0.0
    
    return imc1


def compute_information_measure_correlation_2(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    p_x = stats['p_x']
    p_y = stats['p_y']
    
    # HXY: Entropy of GLCM
    HXY = 0.0
    for i in range(N):
        for j in range(N):
            if normed_glcm[i, j] > 0:
                HXY -= normed_glcm[i, j] * np.log(normed_glcm[i, j])
    
    # HXY2: sum over p_x(i) * p_y(j) * log(p_x(i) * p_y(j))
    HXY2 = 0.0
    for i in range(N):
        for j in range(N):
            if p_x[i] > 0 and p_y[j] > 0:
                HXY2 -= p_x[i] * p_y[j] * np.log(p_x[i] * p_y[j])
    
    # IMC2
    term = 1 - np.exp(-2 * (HXY2 - HXY))
    imc2 = np.sqrt(max(0, term))
    
    return imc2


def compute_inverse_variance(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    
    inv_var = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                inv_var += normed_glcm[i, j] / ((i - j) ** 2)
    
    return inv_var


def compute_difference_average(stats):
    p_diff = stats['p_diff']
    k_values = np.arange(len(p_diff))
    diff_avg = np.sum(k_values * p_diff)
    return diff_avg


def compute_maximal_correlation_coefficient(stats):
    normed_glcm = stats['Pij']
    N = normed_glcm.shape[0]
    p_x = stats['p_x']
    p_y = stats['p_y']
    
    Q = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            sum_k = 0.0
            for k in range(N):
                if p_x[i] > 0 and p_y[k] > 0:
                    sum_k += (normed_glcm[i, k] * normed_glcm[j, k]) / (p_x[i] * p_y[k])
            Q[i, j] = sum_k
    
    eigenvalues = np.linalg.eigvals(Q)
    eigenvalues = np.real(eigenvalues)  # just in case
    eigenvalues_sorted = np.sort(eigenvalues)
    if len(eigenvalues_sorted) > 1:
        mcc = np.sqrt(eigenvalues_sorted[-2]) 
    else:
        mcc = 0.0
    
    return mcc


def compute_maximum_probability(stats):
    normed_glcm = stats['Pij']
    return np.max(normed_glcm)


## Fast Functions (Tester)

@njit(cache=True)
def autocorr_matmul(P):
    N = P.shape[0]
    I = np.arange(1, N + 1, dtype=np.float64)
    J = np.arange(1, N + 1, dtype=np.float64)
    
    return np.sum(I * J * P)


