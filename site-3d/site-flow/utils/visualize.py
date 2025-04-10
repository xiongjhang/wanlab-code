import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.util import get_pic, gaussian_2d, gaussian_fit


def plot_2d_array(array, cmap='viridis', title="2D Array Visualization", xlabel="X-axis", ylabel="Y-axis"):

    if not isinstance(array, np.ndarray) or len(array.shape) != 2:
        raise ValueError("输入必须是二维的 NumPy 数组。")
    
    plt.figure(figsize=(6, 6)) 
    plt.imshow(array, cmap=cmap, interpolation='nearest') 
    plt.colorbar(label='Value') 
    plt.title(title) 
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.show()

def plot_image_with_spots(img, spots, title=None, pred=False):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    clim = tuple(np.percentile(img, (1, 99.8)))
    axs.flat[0].imshow(img, clim=clim, cmap="gray")
    axs.flat[1].imshow(img, clim=clim, cmap="gray")
    axs.flat[1].scatter(spots[:,1], spots[:,0], facecolors='none', edgecolors='orange')
    
    axs.flat[0].axis("off")
    axs.flat[1].axis("off")
    if isinstance(title, str):
        title_subp0 = f"{title}"
        title_subp1 = f"{title} (w/ {'annotation' if not pred else 'prediction'})"
        axs.flat[0].set_title(title_subp0)
        axs.flat[1].set_title(title_subp1)
    return

def plot_intensity(intensity_val: np.ndarray):
    '''Plot intensity of trajectory'''
    pass

def plot_gaussian_3d(offset=0, amp=10, x0=0, y0=0, sigma=2):

    x_range_local = range(0, 7)
    y_range_local = range(0, 7)

    yi, xi = np.meshgrid(y_range_local, x_range_local)
    xyi = np.vstack([xi.ravel(), yi.ravel()])
    
    # 计算高斯函数值
    zi = gaussian_2d(xyi, offset, amp, x0, y0, sigma)
    zi = zi.reshape(xi.shape)  # 重新调整为二维形状
    
    # 创建 3D 图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制高斯曲面
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    # 设置图标题和轴标签
    ax.set_title("2D Gaussian Function")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Amplitude")
    
    # 显示图像
    plt.show()