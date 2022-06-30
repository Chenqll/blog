# General
## baseline
## Graph
### 静态图
### 动态图
## Learning rate scheduler
### reference
- `https://www.cnblogs.com/peachtea/p/13532209.html`
- `https://zhuanlan.zhihu.com/p/520719314`
  
### overview
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
## 使用 numpy 手工实现反向传播
```python

```