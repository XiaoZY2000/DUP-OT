import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

color_list = ["red", "black"]
labels = ["User1", "User2"]

# 高斯分布的参数
# mu_list = [0, 1, 5, 7, 10, 15, 20]
# sigma_list = [1, 1, 1, 1, 1, 1, 1]
mu_list = [1, 5, 7, 17]
sigma_list = [1.2, 1.3, 0.7, 0.9]
w_list = [
    [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2],
    [0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]
]

# 创建横轴（x 值范围足够宽，包含所有分布的中心区域）
x = np.linspace(-3, 30, 500)

# 画图
plt.figure(figsize=(10, 6))
for mu, sigma in zip(mu_list, sigma_list):
    pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, color='blue', label='_nolegend_')

plt.plot([], [], color='blue', label='Gaussian Distributions on Items')

# for i, weights in enumerate(w_list):
#     mixed_pdf = np.zeros_like(x)
#     for mu, sigma, w in zip(mu_list, sigma_list, weights):
#         pdf = norm.pdf(x, mu, sigma)
#         mixed_pdf += w * pdf
#     plt.plot(x, mixed_pdf, color=color_list[i], label=labels[i])

# 添加图例和标签
plt.title("Gaussian Distributions")
plt.xlabel("Item features")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.savefig("Item Distribution 2")