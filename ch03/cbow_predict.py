# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, Softmax

# 样本的上下文数据
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 初始化权重
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 生成层
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)
#ToDo 自己的Softmax
softmax_layer = Softmax()

# 正向传播
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)
prob = softmax_layer.forward(s)

#todo @Toby 自己修改的输出
print("=== 原始得分（logits）===")
print(s)
print("\n=== Softmax 输出（概率分布）===")
print(prob)
print("\n=== 概率总和 =", np.sum(prob))  # 一定 = 1.0
