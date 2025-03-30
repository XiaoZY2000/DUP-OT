import torch
import numpy as np
import ot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def wasserstein_gaussian(mu1, sigma1, mu2, sigma2):
#     """
#     计算两个高斯分布之间的 Wasserstein-2 距离
#     :param mu1: (K, D) 源领域 GMM 组件均值
#     :param sigma1: (K, D, D) 源领域 GMM 组件协方差
#     :param mu2: (K', D) 目标领域 GMM 组件均值
#     :param sigma2: (K', D, D) 目标领域 GMM 组件协方差
#     :return: (K, K') Wasserstein-2 距离矩阵
#     """
#     K, D = mu1.shape
#     K_prime, _ = mu2.shape

#     # 均值距离
#     mu_diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)  # (K, K', D)
#     mean_dist = torch.sum(mu_diff ** 2, dim=-1)  # (K, K')

#     # 协方差距离
#     sigma1_sqrt = torch.linalg.cholesky(sigma1)  # (K, D, D)
#     sigma2_sqrt = torch.linalg.cholesky(sigma2)  # (K', D, D)

#     # sqrt(sigma1) * sigma2 * sqrt(sigma1)
#     mid_term = torch.matmul(sigma1_sqrt, torch.matmul(sigma2, sigma1_sqrt))
#     mid_sqrt = torch.linalg.cholesky(mid_term)  # (K, K', D, D)

#     trace_term = torch.diagonal(sigma1, dim1=-2, dim2=-1).sum(dim=-1).unsqueeze(1) + \
#                  torch.diagonal(sigma2, dim1=-2, dim2=-1).sum(dim=-1).unsqueeze(0) - \
#                  2 * torch.diagonal(mid_sqrt, dim1=-2, dim2=-1).sum(dim=-1)

#     return mean_dist + trace_term  # (K, K')

def wasserstein_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Compute pairwise squared 2-Wasserstein distance between Gaussian components
    mu1: (K, D)
    sigma1: (K, D, D)
    mu2: (K', D)
    sigma2: (K', D, D)
    return: (K, K') distance matrix
    """
    K, D = mu1.shape
    Kp = mu2.shape[0]

    # 均值项距离 (K, K')
    mu_diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)  # (K, K', D)
    mean_dist = torch.sum(mu_diff ** 2, dim=-1)  # (K, K')

    # 计算 sqrt(sigma1)
    sigma1_sqrt = torch.linalg.cholesky(sigma1)  # (K, D, D)

    # 初始化协方差距离矩阵
    trace_term = torch.zeros(K, Kp, device=mu1.device)

    for i in range(K):
        sqrt_sigma1 = sigma1_sqrt[i]  # (D, D)
        for j in range(Kp):
            sigma2_j = sigma2[j]  # (D, D)

            # 计算 sqrt_sigma1 * sigma2 * sqrt_sigma1
            mid = sqrt_sigma1 @ sigma2_j @ sqrt_sigma1.T

            # 取平方根（这里用 Cholesky）
            mid_sqrt = torch.linalg.cholesky(mid)

            # trace 计算
            trace = torch.trace(sigma1[i]) + torch.trace(sigma2[j]) - 2 * torch.trace(mid_sqrt)
            trace_term[i, j] = trace

    return mean_dist + trace_term  # (K, K')

def optimal_transport_matrix(cost_matrix):
    cost_matrix = cost_matrix.cpu().numpy()
    m, n = cost_matrix.shape
    p = np.ones(m) / m
    q = np.ones(n) / n
    print("Training OT Matrix ...")
    T = ot.sinkhorn(p, q, cost_matrix, reg=0.1)
    print("Done")
    
    return torch.tensor(T, dtype=torch.float32).to(device)

def cal_ot_matrix(dataset_pair, result_variable_dict):
    source_domain = dataset_pair[0]
    target_domain = dataset_pair[1]
    mu1 = result_variable_dict[source_domain]["mu_k"]
    mu2 = result_variable_dict[target_domain]["mu_k"]
    sigma1 = result_variable_dict[source_domain]["sigma_k"]
    sigma2 = result_variable_dict[target_domain]["sigma_k"]
    cost_matrix = wasserstein_gaussian(mu1, sigma1, mu2, sigma2)
    ot_matrix = optimal_transport_matrix(cost_matrix)

    return ot_matrix