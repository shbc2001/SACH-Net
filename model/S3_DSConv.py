# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch import nn
import warnings

warnings.filterwarnings("ignore")

"""
This code is mainly the deformation process of our DSConv
"""


class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, extend_scope, morph,
                 if_offset=True):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 4 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(4 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride= (stride *kernel_size, stride *1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(stride *1, stride *kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing,使得摆动方向连续
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph,
                  self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)  # (1,1,63,8)
        if self.morph == 0 or self.morph == 2:  # ！！！！！宽对应列，高对应行！！！！！
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        y_offset, x_offset,w_offset,z_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        # y_center,x_center的shape为(1,9,7,8),延指定方向递增
        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]
            #             print(y_center[0,0,:,:])
            #             print(y_grid[0,0,:,:])
            #             print(x_center[0,0,:,:])
            #             print(x_grid[0,0,:,:])
            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            # print(y_new.shape)
            # print(y_new[0,:,:,0])#第0列所有行上像素点的y方向坐标（morph=0横向卷积，初始y方向无偏移）
            y_offset_new = y_offset.detach().clone()
            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0  # 中心点无偏移
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        elif self.morph == 1:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)
            # print(x_grid)
            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new
        elif self.morph == 2:
            """
            Initialize the kernel and flatten the kernel
                y: num_points//2 ~ -num_points//2
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(
                int(self.num_points // 2),
                -int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            z = torch.linspace(0, 0, 1)

            y, _ = torch.meshgrid(y, z)
            y_spread = y.reshape(-1, 1)
            x, _ = torch.meshgrid(x, z)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)

                z_offset = z_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                x_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (
                                y_offset_new[center + index - 1] + z_offset[center + index] / 2 ** 0.5)
                    y_offset_new[center - index] = (
                                y_offset_new[center - index + 1] + z_offset[center - index] / 2 ** 0.5)

                    x_offset_new[center + index] = (
                                x_offset_new[center + index - 1] + z_offset[center + index] / 2 ** 0.5)
                    x_offset_new[center - index] = (
                                x_offset_new[center - index + 1] + z_offset[center - index] / 2 ** 0.5)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new
        else:
            """
           Initialize the kernel and flatten the kernel
               y: -num_points//2 ~ num_points//2
               x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
               !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
           """
        y = torch.linspace(
            -int(self.num_points // 2),
            int(self.num_points // 2),
            int(self.num_points),
        )
        x = torch.linspace(
            -int(self.num_points // 2),
            int(self.num_points // 2),
            int(self.num_points),
        )
        z = torch.linspace(0, 0, 1)

        y, _ = torch.meshgrid(y, z)
        y_spread = y.reshape(-1, 1)
        x, _ = torch.meshgrid(x, z)
        x_spread = x.reshape(-1, 1)

        y_grid = y_spread.repeat([1, self.width * self.height])
        y_grid = y_grid.reshape([self.num_points, self.width, self.height])
        y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

        x_grid = x_spread.repeat([1, self.width * self.height])
        x_grid = x_grid.reshape([self.num_points, self.width, self.height])
        x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

        y_new = y_center + y_grid
        x_new = x_center + x_grid

        y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
        x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

        y_offset_new = y_offset.detach().clone()
        x_offset_new = x_offset.detach().clone()

        if if_offset:
            y_offset = y_offset.permute(1, 0, 2, 3)
            y_offset_new = y_offset_new.permute(1, 0, 2, 3)

            w_offset = w_offset.permute(1, 0, 2, 3)
            x_offset_new = x_offset_new.permute(1, 0, 2, 3)
            center = int(self.num_points // 2)

            # The center position remains unchanged and the rest of the positions begin to swing
            # This part is quite simple. The main idea is that "offset is an iterative process"
            y_offset_new[center] = 0
            x_offset_new[center] = 0
            for index in range(1, center):
                y_offset_new[center + index] = (y_offset_new[center + index - 1] + w_offset[center + index] / 2 ** 0.5)
                y_offset_new[center - index] = (y_offset_new[center - index + 1] + w_offset[center - index] / 2 ** 0.5)

                x_offset_new[center + index] = (x_offset_new[center + index - 1] + w_offset[center + index] / 2 ** 0.5)
                x_offset_new[center - index] = (x_offset_new[center - index + 1] + w_offset[center - index] / 2 ** 0.5)
            y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
            y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
            x_new = x_new.add(x_offset_new.mul(self.extend_scope))

        y_new = y_new.reshape(
            [self.num_batch, self.num_points, 1, self.width, self.height])
        y_new = y_new.permute(0, 3, 1, 4, 2)
        y_new = y_new.reshape([
            self.num_batch, 1 * self.width, self.num_points * self.height
        ])
        x_new = x_new.reshape(
            [self.num_batch, self.num_points, 1, self.width, self.height])
        x_new = x_new.permute(0, 3, 1, 4, 2)
        x_new = x_new.reshape([
            self.num_batch, 1 * self.width, self.num_points * self.height
        ])
        return y_new, x_new


    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """


    def _bilinear_interpolate_3D(self, input_feature, y, x):
        # print(y[0,:,0])
        # print(x[0,:,7])
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()
        zero = torch.zeros([]).int()
        max_y = self.width - 1  # 0~max_y
        max_x = self.height - 1  # 0~max_x

        # find 8 grid locations
        # print(y[:10])
        y0 = torch.floor(y).int()  # 向下取整（top)
        # print(y0[:10])
        y1 = y0 + 1  # (bottom)
        # print(x)
        x0 = torch.floor(x).int()  # left
        # print(x0)
        x1 = x0 + 1  # right

        # clip out coordinates exceeding feature map volume
        # print('max_y=',max_y)
        # print(y0)

        y0 = torch.clamp(y0, zero, max_y)
        # print(y0)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        # print(input_feature_flat.shape)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)  # shape=(1,7,8,c)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)  # shape=(56,c)
        # print(input_feature_flat.shape)
        dimension = self.height * self.width  # 56维向量

        base = torch.arange(self.num_batch) * dimension
        # print(base.shape)
        base = base.reshape([-1, 1]).float()  # (1,1)
        # print(base.shape)
        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()  # (1,56*k)
        # print(repeat.shape)
        base = torch.matmul(base, repeat)  # (1,56*k)
        # print(base.shape)
        base = base.reshape([-1])  # (56*k)
        # print(base)#全0
        base = base.to(self.device)

        base_y0 = base + y0 * self.height  # 因为input_feature_flat的shape是(1,7,8,c)，矩阵展平
        base_y1 = base + y1 * self.height
        # 所有点(7*8*k)的临近坐标
        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0  # 左上
        index_c0 = base_y0 - base + x1  # 右上

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0  # 左下
        index_c1 = base_y1 - base + x1  # 右下
        # print(index_a0)
        # get 8 grid values，所有点的临近值
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)
        # print(value_a0.shape)
        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        # 双线性插值
        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0 or self.morph == 2:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
            # print(outputs.shape)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs


    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature


# Code for testing the DSConv
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = np.random.rand(1, 3, 7, 8)
    A = A.astype(dtype=np.float32)
    A = torch.from_numpy(A)
    conv0 = DSConv(
        in_ch=3,
        out_ch=10,
        kernel_size=9,
        extend_scope=1,
        morph=0,
        if_offset=True,
        device=device)
    if torch.cuda.is_available():
        A = A.to(device)
        conv0 = conv0.to(device)
    out = conv0(A)
    print(out.shape)
    # print(out)