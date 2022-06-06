# 统计学习
## 第一章 统计学习及监督学习概论
### 统计学习的分类
#### 监督学习
- 本质是学习输入到输出的映射规律，该映射由模型来表示，最好的模型属于映射空间，该空间也成为假设空间。
- 假设输入与输出的随机变量X，Y，遵循联合概率分布*P(X,Y)*，预测可写作*P(y|x)*,or *y=f(x)*
#### 无监督学习
- 本质是学习数据中心的统计规律或潜在结构
#### 强化学习
#### 半监督与主动学习

#### 贝叶斯学习
- 本质是利用贝叶斯定理，计算在给定数据条件下模型的条件概率，即后验概率
- **贝叶斯估计和极大似然估计在思想上有很大的不同，代表统计学中贝叶斯学派和统计学派对统计的不同认识？？？** 假设先验分布是均匀分布，去后验概率最大，就能从贝叶斯估计得到极大似然估计
- 上一点不懂的可参照 <https://www.cnblogs.com/sylvanas2012/p/5058065.html>

#### 核方法
- 线性到非线性：求映射-->求映射后的内积。
- 核函数直接定义内积，完成映射

### 统计学习方法三要素
#### 模型
- 所要学习的条件概率分布或者决策函数，即确定假设空间。
#### 策略
- 按照怎样的准则在假设空间中选择最优模型
- 损失函数--只能得到经验损失，样本量小时，经验损失不能代表真正的期望风险，使用正则化对经验损失进行优化=结构风险：
    · 过拟合 训练误差逐渐减小然而测试误差将先减小后增大
    · 正则化包括两项，一个函数为模型复杂度，另一个是超参数，主要是为了缓解过拟合（模型越复杂，正则化项越大，对于损失函数的惩罚也就越大，可以达到由于模型复杂度高而造成的过拟合现象）
    · 交叉验证 将数据分为训练、验证、测试集
#### 算法

#### 可参考资料
<https://zhuanlan.zhihu.com/p/35141478>

# Learning rate scheduler
## reference
- `https://www.cnblogs.com/peachtea/p/13532209.html`
- `https://zhuanlan.zhihu.com/p/520719314`
  
## overview
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

