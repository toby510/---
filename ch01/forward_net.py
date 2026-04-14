# coding: utf-8
import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out

#自己新增的，源码里没有
class Softmax:
    def __init__(self):
        self.params = []

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # 防溢出
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
            #自己新增的，用于输出概率
            Softmax()
        ]

        # 将所有的权重整理到列表中
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
prob = model.predict(x)
pred_class = np.argmax(prob, axis=1) # 输出最终类别

print("=== 每个类别的概率（0~1） ===")
print(prob)
print("\n=== 最终分类结果（0/1/2） ===")
print(pred_class)
