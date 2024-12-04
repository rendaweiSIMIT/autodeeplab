import torch
import torch.nn as nn
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                # 为池化操作添加 BatchNorm 层
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        # 检查输入是否为 None，避免非法调用
        if x is None:
            raise ValueError("Input tensor x is None")
        # 权重与操作的加权和
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate):
        super(Cell, self).__init__()
        self.C_out = C

        # 如果有上一个单元的输出，进行预处理
        if C_prev_prev != -1:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        # 根据 rate 决定预处理方式
        if rate == 2:
            self.preprocess1 = FactorizedReduce(C_prev, C, affine=False)
        elif rate == 0:
            self.preprocess1 = FactorizedIncrease(C_prev, C)
        else:
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()

        # 初始化操作
        for i in range(self._steps):
            for j in range(2 + i):
                if C_prev_prev != -1 and j != 0:
                    op = MixedOp(C, stride=1)
                else:
                    op = nn.Identity()  # 使用 nn.Identity 替代 None
                self._ops.append(op)

        # 用于最终输出的处理
        self.ReLUConvBN = ReLUConvBN(self._multiplier * self.C_out, self.C_out, 1, 1, 0)

    def forward(self, s0, s1, weights):
        # 预处理前两个输入
        if s0 is not None:
            s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]  # 初始化状态列表
        offset = 0  # 记录操作偏移量

        for i in range(self._steps):
            print(f"Step {i}: States: {states}, Ops: {self._ops[offset:offset + len(states)]}")

            s = sum(
                self._ops[offset + j](h) if isinstance(self._ops[offset + j], nn.Identity) 
                else self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
                if h is not None and isinstance(self._ops[offset + j], nn.Module)
            )
            offset += len(states)  # 更新偏移量
            states.append(s)  # 添加新状态

        # 拼接最后几步的输出特征图
        concat_feature = torch.cat(states[-self._multiplier:], dim=1)
        return self.ReLUConvBN(concat_feature)
