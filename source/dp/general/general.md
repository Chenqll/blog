# General
## 1.baseline
### 1.1 reference
https://docs.oneflow.org/master/basics/01_quickstart.html
### 1.2 General
#### 1.2.1 quick start
加载数据-->搭建网络-->训练模型-->保存模型
- 加载数据 dataset 和 DataLoader。
  - 可以在 flowvision 工具包中找到常用的数据集 dataset 并加载
  - 使用 `flow.utils.data.dataloader` 类可以将 dataset 封装为迭代器
- 搭建网络需要继承 `nn.Module` 模块，并重写 `__init__` 和 `forward` 方法。
- 训练模型分四步
  - 定义损失函数和优化器optimizer
  - 定义 `train()` 函数，实现前向传播 `pred=model(x)`；计算损失 `loss=loss_fn(pred,y)`；反向传播 `optimizer.zero_grad()  loss.backward()`；更新参数 `optimizer.step()`
#### 1.2.2 Tensor
Tensor 是多维数组 or 矩阵，神经网络使用矩阵计算可以大大加快计算速度，oneflow中提供很多操作 tensor 的算子， Tensor 和 算子 op 构成神经网络
- tensor 与普通多维数组的区别是可以运行在 GPU 中，并且 oneflow 为其提供了自动求导的功能
- 构建 Tensor 有三种方法：直接从数据构建；通过 Numpy 数组构建；使用算子构建
#### 1.2.3 搭建神经网络
- `oneflow.nn` 接口下提供了构建神经网络所需的常见 Module。大部分网络都需要继承 nn.Module 再复写 `__init__()` 和 `forward()`
- nn.functional 接口下也提供了构建神经网络所需要的 API，但是与nn下的接口作为类接口出现不同，nn.functional 下的接口是作为函数接口出现的，可以理解为nn.functional 封装后就变成了 nn 模块下的接口。
#### 1.2.4 自动求导 反向传播
- 已知计算图的计算过程 ： 前向传播--》损失计算---》反向传播(由于目标是要使得损失函数最小，而损失函数可以看成是与 w,b 相关的二元函数，通过调整 w,b 的取值，使得loss 最小，则需要求得 loss 分别对 w,b 的梯度，通过梯度的方向不断优化 w，b 直到找到能使得loss 最小的值)
- grad 是计算图中每一个结点的属性，可以通过 `require_grad()` 属性开关是否保留梯度，如果设置该参数为 `Ture`，则在 loss.backward() 时自动计算结点的 grad
#### 1.2.5 使用 Numpy 构建一个网络
```python
import numpy as np

# 定义超参数
ITER_COUNT=500
LR=0.01

# 前向传播
def forward(x,w):
  return np.matmul(x,w)


# 损失函数
def loss(y_pred,y):
  return(y_pred-y)**2.sum()

# 计算梯度，使用上述的损失函数，于是得出下述计算梯度的方式
def gradient(x,y,y_pred):
  return np.matmul(x.T,2*(y_pred-y))

if __name__=="__main__":
  x=np.array([[1,2],[2,3],[3,4],[4,5]],dtype=np.float32)
  y=np.array([[8],[13],[26],[9]],dtype=np.float32)

  w=np.array([[2],[1]],dtype=np.float32)

  for i in range(0,ITER_COUNT):
    y_pred=forward(x,w)
    l=loss(y_pred,y)
    

    grad=gradient(x,y,y_pred)
    # SGD 的更新参数的优化器方法
    w-=LR*grad
```

#### 1.2.6 使用包装好的API实现 1.2.5
```python
import oneflow as flow

x = flow.tensor([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=flow.float32)
y = flow.tensor([[8], [13], [26], [9]], dtype=flow.float32)

# 封装为一个复杂的 Module，需要继承 nn.Module
# 完成神经网络的前向传播过程
class MyLrModule(flow.nn.Module):
  def __init__(self,lr,iter_count):
    super.__init__()
    self.w=flow.nn.Parameter(flow.tensor([[2],[1]],dtype=flow.float32))
    self.lr=lr
    self.iter_count=iter_count
  
  def forward(self,x):
    return flow.matmul(x,self.w)

# 损失函数
loss=flow.nn.MSELoss(reduction="sum")

# 计算梯度
# 需要使用optimizer在计算图中自动计算梯度
# 调用backward 函数时，会将每个叶子结点的 grad 算出来，之后调用optimizer.step(),进行参数更新
optimizer=flow.optim.SGD(model.parameters(),model.lr)

for i in range (0,model.iter_count):
  y_pred=model(x)
  l=loss(y_pred,y)

  opimizer.zero_grad()
  l.backward()
  optimizer.step()

```
## 2.Graph
### 2.1静态图
### 2.2动态图
## 3.Learning rate scheduler
### 3.1reference
- `https://www.cnblogs.com/peachtea/p/13532209.html`
- `https://zhuanlan.zhihu.com/p/520719314`
  
### 3.2overview
- lr --> optimizer
- Pytorch 有6种学习率调整策略，都继承自 `class_LRScheduler` 
  ```python
    class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        ....
        ....
        主要得到 optimizer ，last_epoch 参数
  ```
- 以 `StepLR` - 等间隔调整学习率 为例子
    ```python
    optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) #调用学习率策略 StepLR
    # 此时 lr_scheduler 与 optimizer 相关联
    ```
  ```python
    import torch
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import pdb
    # //设置初始学习率
    LR = 0.1        
    iteration = 10
    max_epoch = 200 
    # --------- fake data and optimizer  ---------

    weights = torch.randn((1), requires_grad=True)
    # weights==tensor([0.7652], requires_grad=True)
    pdb.set_trace()

    target = torch.zeros((1))
    # target==tensor([0.])

    # //构建虚拟优化器，为了 lr_scheduler关联优化器 
    optimizer = optim.SGD([weights], lr=LR, momentum=0.9)

    # ---------------- 1 Step LR --------
    # flag = 0
    flag = 1
    if flag:
    # //设置optimizer、step_size等间隔数量：多少个epoch之后就更新学习率lr、gamma
        scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略

        lr_list, epoch_list = list(), list()
        for epoch in range(max_epoch):

            lr_list.append(scheduler_lr.get_last_lr())
            epoch_list.append(epoch)

            for i in range(iteration):

                loss = torch.pow((weights - target), 2)
                loss.backward()
    # //优化器参数更新
                optimizer.step()
                optimizer.zero_grad()
    #//学习率更新
            scheduler_lr.step()

        plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.legend()
        plt.show()

  ```
## 4.使用 numpy 手工实现反向传播
```python

```