import numpy as np
from numba import njit
from skimage.feature import graycomatrix



@njit(cache=True)
def get_p_x(Pij):
    N = Pij.shape[0]
    p_x = np.zeros(N)
    for i in range(N):
        for j in range(N):
            p_x[i] += Pij[i, j]
    return p_x

@njit(cache=True)
def get_p_y(Pij):
    N = Pij.shape[0]
    p_y = np.zeros(N)
    for j in range(N):
        for i in range(N):
            p_y[j] += Pij[i, j]
    return p_y

@njit(cache=True)
def get_p_sum(Pij):
    N = Pij.shape[0]
    p_sum = np.zeros((2 * N) - 1)
    for i in range(N):
        for j in range(N):
            p_sum[i + j] += Pij[i, j]
    return p_sum

@njit(cache=True)
def get_p_diff(Pij):
    N = Pij.shape[0]
    p_diff = np.zeros(N)
    for i in range(N):
        for j in range(N):
            p_diff[np.abs(i - j)] += Pij[i, j]
    return p_diff

@njit(cache=True)
def get_mu_x(Pij):
    N = Pij.shape[0]
    p_x = get_p_x(Pij)
    mu_x = 0.0
    for i in range(N):
        mu_x += i * p_x[i]
    return mu_x

@njit(cache=True)
def get_mu_y(Pij):
    N = Pij.shape[0]
    p_y = get_p_y(Pij)
    mu_y = 0.0
    for j in range(N):
        mu_y += j * p_y[j]
    return mu_y

@njit(cache=True)
def get_mu_sum(Pij):
    N = Pij.shape[0]
    p_sum = get_p_sum(Pij)
    mu_sum = 0.0
    for k in range(len(p_sum)):
        mu_sum += (k+2) * p_sum[k]
    return mu_sum

@njit(cache=True)
def get_mu_diff(Pij):
    N = Pij.shape[0]
    p_diff = get_p_diff(Pij)
    mu_diff = 0.0
    for k in range(N):
        mu_diff += k * p_diff[k]
    return mu_diff

# @njit(cache=True)
# def get_sigma_x(Pij):
#     N = Pij.shape[0]
#     p_x = get_p_x(Pij)
#     mu_x = get_mu_x(Pij)
#     sigma_x = 0.0
#     for i in range(N):
#         sigma_x += ((i - mu_x) ** 2) * p_x[i]
#     return np.sqrt(sigma_x)

# @njit(cache=True)
# def get_sigma_y(Pij):
#     N = Pij.shape[0]
#     p_y = get_p_y(Pij)
#     mu_y = get_mu_y(Pij)
#     sigma_y = 0.0
#     for j in range(N):
#         sigma_y += ((j - mu_y) ** 2) * p_y[j]
#     return np.sqrt(sigma_y)

@njit(cache=True)
def get_HX(Pij):
    N = Pij.shape[0]
    p_x = get_p_x(Pij)
    HX = 0.0
    for i in range(N):
        if p_x[i] > 0.0:
            HX -= p_x[i] * np.log(p_x[i])
    return HX

@njit(cache=True)
def get_HY(Pij):
    N = Pij.shape[0]
    p_y = get_p_y(Pij)
    HY = 0.0
    for j in range(N):
        if p_y[j] > 0.0:
            HY -= p_y[j] * np.log(p_y[j])
    return HY

@njit(cache=True)
def get_HXY(Pij):
    N = Pij.shape[0]
    HXY = 0.0
    for i in range(N):
        for j in range(N):
            if Pij[i, j] > 0.0:
                HXY -= Pij[i, j] * np.log(Pij[i, j])
    return HXY

@njit(cache=True)
def get_HXY1(Pij):
    N = Pij.shape[0]
    p_x = get_p_x(Pij)
    p_y = get_p_y(Pij)
    HXY1 = 0.0
    for i in range(N):
        for j in range(N):
            if p_x[i] > 0.0 and p_y[j] > 0.0:
                HXY1 -= Pij[i, j] * np.log(p_x[i] * p_y[j])
    return HXY1

@njit(cache=True)
def get_HXY2(Pij):
    N = Pij.shape[0]
    p_x = get_p_x(Pij)
    p_y = get_p_y(Pij)
    HXY2 = 0.0
    for i in range(N):
        for j in range(N):
            if p_x[i] > 0.0 and p_y[j] > 0.0:
                HXY2 -= p_x[i] * p_y[j] * np.log(p_x[i] * p_y[j])
    return HXY2





# ---------------------------------------------------------------------------
# Features 
# ---------------------------------------------------------------------------

@njit(cache=True)
def compute_autocorrelation(Pij):
    N = Pij.shape[0]
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (i + 1) * (j + 1) * Pij[i, j]
    return val


@njit(cache=True)
def compute_cluster_prominence(Pij):
    N = Pij.shape[0]
    mu_x = get_mu_x(Pij)
    mu_y = get_mu_y(Pij)
    mu = (mu_x + mu_y)/2
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (((i + 1) + (j + 1) - (2 * mu) ) ** 4) * Pij[i, j]
    return val


@njit(cache=True)
def compute_cluster_6(Pij):
    N = Pij.shape[0]
    mu_x = get_mu_x(Pij)
    mu_y = get_mu_y(Pij)
    mu = (mu_x + mu_y)/2
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (((i + 1) + (j + 1) - (2 * mu) ) ** 6) * Pij[i, j]
    return val

@njit(cache=True)
def compute_cluster_7(Pij):
    N = Pij.shape[0]
    mu_x = get_mu_x(Pij)
    mu_y = get_mu_y(Pij)
    mu = (mu_x + mu_y)/2
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (((i + 1) + (j + 1) - (2 * mu) ) ** 7) * Pij[i, j]
    return val

@njit(cache=True)
def compute_cluster_8(Pij):
    N = Pij.shape[0]
    mu_x = get_mu_x(Pij)
    mu_y = get_mu_y(Pij)
    mu = (mu_x + mu_y)/2
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (((i + 1) + (j + 1) - (2 * mu) ) ** 8) * Pij[i, j]
    return val

@njit(cache=True)
def compute_cluster_9(Pij):
    N = Pij.shape[0]
    mu_x = get_mu_x(Pij)
    mu_y = get_mu_y(Pij)
    mu = (mu_x + mu_y)/2
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (((i + 1) + (j + 1) - (2 * mu) ) ** 9) * Pij[i, j]
    return val

@njit(cache=True)
def compute_cluster_10(Pij):
    N = Pij.shape[0]
    mu_x = get_mu_x(Pij)
    mu_y = get_mu_y(Pij)
    mu = (mu_x + mu_y)/2
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (((i + 1) + (j + 1) - (2 * mu) ) ** 10) * Pij[i, j]
    return val



@njit(cache=True)
def compute_cluster_shade(Pij):
    N = Pij.shape[0]
    mu_x = get_mu_x(Pij)
    mu_y = get_mu_y(Pij)
    mu = (mu_x + mu_y)/2
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (((i + 1) + (j + 1) - (2 * mu) ) ** 3) * Pij[i, j]
    return val


@njit(cache=True)
def compute_cluster_tendency(Pij):
    N = Pij.shape[0]
    mu_x = get_mu_x(Pij)
    mu_y = get_mu_y(Pij)
    mu = (mu_x + mu_y)/2
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (((i + 1) + (j + 1) - (2 * mu) ) ** 2) * Pij[i, j]
    return val


@njit(cache=True)
def compute_difference_average(Pij):
    p_diff = get_p_diff(Pij)
    val = 0.0
    for k in range(len(p_diff)):
        val += k * p_diff[k]
    return val


@njit(cache=True)
def compute_difference_entropy(Pij):
    N = Pij.shape[0]
    p_diff = get_p_diff(Pij)
    val = 0.0
    for k in range(N):
        if p_diff[k] > 0.0:
            val -= p_diff[k] * np.log(p_diff[k])
        else: val -= 0
    return val


@njit(cache=True)
def compute_difference_variance(Pij):
    N = Pij.shape[0]
    p_diff = get_p_diff(Pij)
    mu_diff = get_mu_diff(Pij)
    val = 0.0
    for k in range(N):
        val += ((k - mu_diff) ** 2) * p_diff[k]
    return val


@njit(cache=True)
def compute_entropy(Pij):
    return get_HXY(Pij)


@njit(cache=True)
def compute_information_measure_correlation_1(Pij):
    N = Pij.shape[0]
    HX = get_HX(Pij)
    HY = get_HY(Pij)
    HXY = get_HXY(Pij)
    HXY1 = get_HXY1(Pij)
    val = 0.0
    if max(HX, HY) > 0.0:
        val = (HXY - HXY1) / max(HX, HY)
    else: val = 0.0
    return val


@njit(cache=True)
def compute_information_measure_correlation_2(Pij):
    N = Pij.shape[0]
    HXY = get_HXY(Pij)
    HXY2 = get_HXY2(Pij)
    val = 1.0 - np.exp(-2.0 * (HXY2 - HXY))
    if val < 0.0:
        val = 0.0
    return val ** 0.5


@njit(cache=True)
def compute_inverse_difference(Pij):
    N = Pij.shape[0]
    val = 0.0
    for i in range(N):
        for j in range(N):
            d = np.abs(i-j)
            val += Pij[i, j] / (1.0 + d)
    return val


@njit(cache=True)
def compute_inverse_difference_norm(Pij):
    N = Pij.shape[0]
    val = 0.0
    for i in range(N):
        for j in range(N):
            d = np.abs(i-j)
            val += Pij[i, j] / (1.0 + (d/N))
    return val


@njit(cache=True)
def compute_inverse_difference_moment(Pij):
    N = Pij.shape[0]
    val = 0.0
    for i in range(N):
        for j in range(N):
            d = (i - j) ** 2
            val += Pij[i, j] / (1.0 + d)
    return val


@njit(cache=True)
def compute_inverse_difference_moment_norm(Pij):
    N = Pij.shape[0]
    val = 0.0
    for i in range(N):
        for j in range(N):
            d = (i - j) ** 2
            val += Pij[i, j] / (1.0 + (d/N))
    return val


@njit(cache=True)
def compute_inverse_variance(Pij):
    N = Pij.shape[0]
    val = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                val += Pij[i, j] / ((i - j) ** 2)
    return val


@njit(cache=True)
def compute_joint_average(Pij):
    N = Pij.shape[0]
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (i + 1) * Pij[i, j]
    return val


@njit(cache=True)
def compute_maximum_probability(Pij):
    return np.max(Pij)


@njit(cache=True)
def compute_sum_average(Pij):
    N = Pij.shape[0]
    p_sum = get_p_sum(Pij)
    val = 0.0
    for k in range(len(p_sum)):
        val += (k+2) * p_sum[k]
    return val


@njit(cache=True)
def compute_sum_entropy(Pij):
    N = Pij.shape[0]
    p_sum = get_p_sum(Pij)
    val = 0.0
    for k in range(len(p_sum)):
        if p_sum[k] > 0.0:
            val -= p_sum[k] * np.log(p_sum[k])
    return val


@njit(cache=True)
def compute_sum_of_squares(Pij):
    N = Pij.shape[0]
    mu_x = get_mu_x(Pij)
    val = 0.0
    for i in range(N):
        for j in range(N):
            val += (((i + 1) - mu_x) ** 2) * Pij[i, j]
    return val


@njit(cache=True)
def compute_sum_variance(Pij):
    N = Pij.shape[0]
    p_sum = get_p_sum(Pij)
    mu_sum = get_mu_sum(Pij)
    val = 0.0
    for k in range(len(p_sum)):
        val += (((k+2) - mu_sum) ** 2) * p_sum[k]
    return val


@njit(cache=True)
def compute_maximal_correlation_coefficient(Pij):
    N = Pij.shape[0]
    p_x = get_p_x(Pij)
    p_y = get_p_y(Pij)

    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(N):
                if p_x[i] > 0.0 and p_y[k] > 0.0:
                    s += (Pij[i, k] * Pij[j, k]) / (p_x[i] * p_y[k])
            Q[i, j] = s
    U, S, Vh = np.linalg.svd(Q)
    return S[1] if len(S) > 1 else 0.0


