#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: 徐聪
# datetime: 2022-10-23 11:38
# software: PyCharm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from BP import BPNet
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_path = r"实验2数据集/winequality_data.xlsx"

# 读取数据
df = pd.read_excel(data_path)
# 获取label
labels = list(set(df["quality label"]))
# 建立label到id的映射
id2label = {}
label2id = {}
cnt = 0
for label in labels:
    id2label[cnt] = label
    label2id[label] = cnt
    cnt += 1

# 将样本标签
y = [label2id[label] for label in list(df["quality label"])]
y = np.array(y).reshape((len(y), 1))
x = np.array(df.iloc[:, 0:11])

# 归一化处理
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 训练集和测试集切分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 训练模型
bp = BPNet(11, 100, 3)
erro_list, acc_list, epoch_list = bp.train(x_train, y_train, 0.1, 500)

# 绘制图像
plt.figure(1)
plt.plot(epoch_list, erro_list, "r-", label="loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Change of loss function with training epochs")
plt.legend()
plt.grid()
plt.show()

plt.figure(1)
plt.plot(epoch_list, acc_list, "b-", label="acc")
plt.xlabel("epochs")
plt.title("Change of accuracy with training epochs")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.show()

# 模型准确性分析
print("神经网络模型在训练集上的准确性分析")
print(classification_report(y_train, bp.query(x_train)))
print("神经网络模型在测试集上的准确性分析")
print(classification_report(y_test, bp.query(x_test)))

