import matplotlib.pyplot as plt
import numpy as np
def save_map(x,name,channel):
    # 假设 x 是一个 PyTorch Tensor，你可以将其转换为 NumPy 数组
    x_np = x.detach().cpu().numpy()

    # 假设 x_np 的形状是 (batch_size, channels, height, width)，选择一个示例
    single_example = x_np[0]  # 选择第一个样本的特征图
    average_feature_map = np.mean(single_example, axis=0)
    # 选择一个通道，这里选择第一个通道
    # channel_to_visualize = channel
    # feature_map = single_example[channel_to_visualize, :, :]

    # 使用 matplotlib 显示特征图
    # feature_map[feature_map>0.1]=1
    # feature_map[feature_map <0.1] = 0
    plt.imshow(average_feature_map, cmap='gray')  # 选择合适的颜色映射
    # plt.colorbar()  # 添加颜色条，显示数值与颜色的对应关系
    # plt.title(f'Feature Map for Channel {channel_to_visualize}')
    plt.savefig('model/23xymax/'+name+'.png')
