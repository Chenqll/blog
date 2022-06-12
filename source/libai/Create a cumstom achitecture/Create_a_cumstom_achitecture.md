# Create a cumstom achitecture
较为复杂的深度学习项目一般都有大量的可配置参数，为了能够更加方便灵活地配置各种参数，LiBai 借鉴了detectron2 的 lazy config system 。可以通过单个配置文件来轻松配置各种参数。配置文件本质上是 Python 代码文件，因此可以添加任意的逻辑，以及通过导入其他的配置文件来复用配置项。
LiBai 定义了一组必需的配置字段，包括：

+ model: 使用的模型的相关配置

+ graph: 有关 OneFlow 中的 静态图模式(nn.Graph) 的配置项

+ train： 训练与评估相关的配置

+ optim: 优化算法相关的配置

+ dataloader: 数据集与 data loader 相关的配置

LiBai 在 configs/common 中提供了以上配置字段的默认值，可以在编写配置文件时导入，然后进行部分修改。但是想要更多地控制特定模型参数的用户可以从几个基类创建一个自定义的 libai 模型。这对于任何有兴趣学习、训练或试验 libai 模型的人来说可能特别有用。在本指南中，我们将以一个简易的 MNIST 手写数字识别项目为例，深入了解如何创建一个libai 项目,了解如何：

+ [加载和自定义配置](#配置)
+ [定义和配置 Model](#定义和配置+model)
+ [定义和配置 Dataset 与 DataLoader](#定义和配置-dataset-与-dataloader)
+ 训练过程配置
  
## 配置
## 定义和配置 Model
model 是神经网络模型的实例，也是全局配置文件 `config.py` 的必需元素之一。

使用 LiBai 训练网络模型，我们首先要定义一个 model 类，然后在 `config.py` 中 import 并声明。

### 定义 Model

我们将模型定义在 model.py 文件中：

```python
# modeling/model.py
import oneflow.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes = 10):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self.loss_func = nn.CrossEntropyLoss()
    def forward(self, inputs, labels):
        inputs = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs)
        if labels is not None and self.training:
            losses = self.loss_func(logits, labels)
            return {"losses": losses}
        else:
            return {"prediction_scores": logits}
```
我们需要定义 `__init__` 和 `forward(...)` 2 个方法，其中 `__init__` 接收初始化参数，完成对计算单元和损失函数的初始化；`forward(...)` 接收 input 等数据完成前向计算，返回 logits 或 loss。
需要遵守如下规定：

+ model 类必须继承自 `oneflow.nn.Module`
+ `__init__` 内将 `损失函数` 初始化为成员变量。如`self.loss_func = nn.CrossEntropyLoss()` ，原因是 Libai 要求在模型内部计算 loss 。
+ `forward(...)` 所接收的参数来自于 DataSet 类 `__getitem__` 函数的返回值（下一节会讲到）。本例采用一个图片分类模型，故接收 images 和 labels 两个参数。对于 transformers 系列模型一般需接收 `input_id`, `attention_mask`, `tokentype_id` 等参数。
+ `forward(...)` 内部需添加当前处于 training 或 eval 阶段的判断，若在 training 阶段则需计算 loss 并返回。
+ `forward(...)` 的返回类型是一个 dict，在 training 期间返回 loss ，在 inference 期间返回 prediction scores。


### 配置 Model

在 `config.py` 中配置 model ，首先需要用 `from modeling.model import NeuralNetwork` 将模型类导入，再用 `model = LazyCall(类名)(参数)` 将其声明。

```python
from libai.config import LazyCall
from modeling.model import NeuralNetwork
...
model = LazyCall(NeuralNetwork)(num_classes=10)
...
```
**注意**

+ 在全局配置文件中使用导入的 Model 类时必须将其命名为 model。
+ 一般通过 `model = LazyCall(类名)(参数1= ... , 参数2 = ... )` 的方式给模型`__init__(...)`传参。另一种方式是为模型的 `__init__` 函数添加 `@configurable` 标签，使其接收一个 dict 类型的配置参数，示例如下：

```python
# modeling/model.py
from libai.config import configurable
class NeuralNetwork(nn.Module):
    @configurable
    def __init__(self, num_classes=10):
        # ...
# configs/config.py
# ...
cfg={
	num_classes=10
}
model = LazyCall(NeuralNetwork)(cfg)
# ...
```

## 定义和配置 Dataset 与 DataLoader

### 定义 Dataset 类

我们需要定义一个 和 PyTorch Dataset 风格相似的 Dataset 类，用于完成数据读取、预处理和按索引读取功能。

与一般的 PyTorch Dataset 不同，我们需要在 `__getitem__` 中对 tensor 进行封装，最终返回一个 `Instance` 类型的对象，而不是返回 Tensor Tuple 或其他类型。这样是为了更方便地进行分布式训练。

```python
# dataset/dataset.py
import oneflow as flow
from flowvision import transforms
from flowvision import datasets
from oneflow.utils.data import Dataset
from libai.data.structures import DistTensorData, Instance
class MnistDataSet(Dataset):
    def __init__(self, path, is_train):
        self.data = datasets.MNIST(
            root=path,
            train=is_train,
            transform=transforms.ToTensor(),
            download=True,
            source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/"
        )
    def __getitem__(self, idx):
        sample = Instance(
            inputs=DistTensorData(
                flow.tensor(self.data[idx][0].clone().detach(), dtype=flow.float32)
            ),
            labels=DistTensorData(
                flow.tensor(self.data[idx][1], dtype=flow.int),
                placement_idx=-1)
        )
        return sample
    def __len__(self):
        return len(self.data)
```
+ Dataset 类必须继承自 `oneflow.utils.data.Dataset`

+ `__getitem__` 方法接收一个 int 类型的索引，返回对应的样本。Libai 提供了 `DistTensorData` 和 `Instance` 两个类，用于封装普通的 `oneflow.Tensor`，需要完成以下 4 个步骤：

    1. 将原始数据转换为 `oneflow.Tensor` 类型
    2. 将 `oneflow.Tensor`  转换为 `DistTensorData` 类型。在获取一个批量的数据时，`DistTensorData`内部会调用 `to_global()` 将 local tensor 转化为 global tensor
    3. 使用关键字参数的形式将若干个 `DistTensorData` 包装为一个 `Instance` 。`Instance` 对象会将这些 `DistTensorData` 对象存储为自身的属性。需要特别注意的是，**该 Instance 对象的关键字参数名必须与模型类的 forward 方法的参数名相同**
    4. 返回 `Instance` 对象

+ 在`__len__` 方法中返回样本总数



  **注意 ：**若某个 tensor 不参与前向传播（例如只需要传递给损失函数计算 loss），则可以在创建 `DistTensorData` 时传入`placement_idx=-1` ，便于配置流水并行训练。

### 创建 DataLoader 对象

在全局 `config.py` 文件中，首先通过 `from dataset.dataset import MnistDataSet` 导入自定义 Dataset 类，然后创建 DataLoader   对象。

```python
# configs/config.py
from omegaconf import OmegaConf
from libai.config import LazyCall
from libai.data.build import build_image_train_loader, build_image_test_loader
from dataset.dataset import MnistDataSet
dataloader = OmegaConf.create()
dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(MnistDataSet)(
            path="/workspace/quickstart/data/",
            is_train=True,
        )
    ],
    num_workers=4,
)
dataloader.test = [LazyCall(build_image_test_loader)(
    dataset=LazyCall(MnistDataSet)(
        path="/workspace/quickstart/data/",
        is_train=False,
    ),
    num_workers=4,
)]
```

+ OmegaConf 是一个分层配置系统，使用 `dataloader = OmegaConf.create()` 获取一个配置对象来配置 DataLoader。

+ dataloader.train 对应训练集，dataloader.test 对应测试集。

+ `build_image_train_loader` 和 `build_image_test_loader` 是 LiBai 提供的用于创建图像数据的训练集和测试集 DataLoader 的两个函数，便于直接使用 Dataset 对象创建 DataLoader 对象。此外还有 `build_nlp_train_loader`、`build_nlp_test_loader` 等。这些函数的 `dataset` 参数接受一个 Dataset 对象或者 Dataset 对象组成的列表。

## 训练过程配置

如前所述，"train" 是全局配置文件中的必需字段，涵盖了与训练与评估有关的配置，主要包含以下 5 个方面：

+ 训练过程中使用的超参数，包括 learning rate、batch size、epoch、iteration 、warmup_ratio 等
+ AMP、Activation Checkpointing、nccl fusion 、zero_optimization 等选项，使用 `dict(enabled=True/False)` 来开启/关闭。
+ 分布式训练相关配置，使用 `dist = dict( ... ) ` 设置。
+ libai.scheduler 相关参数
+ evaluation 相关参数


我们既可以**在一个单独的 train.py 中配置好全部参数，然后导入**，也可以**先在 config.py 中导入 Libai 默认配置，然后用 `train.update(...)` 修改相关字段**。

本项目采用如下方式配置 train：

```python
# configs/config.py
from configs.train import train
train.update(
    dict(
        recompute_grad=dict(enabled=True),   # 计算梯度
        amp=dict(enabled=True),              # 启动 AMP 自动混合精度进行训练（不影响推理性能）。
        output_dir="output/MNIST/",          # 指定 checkpoint 的输出路径
        train_micro_batch_size=128,          # train & test 阶段的 batch size 
        test_micro_batch_size=32,
        train_epoch=20,                      # 执行 20 轮 训练
        train_iter=0,
        eval_period=100,                     # 每经过 100 次训练迭代，就执行一次 eval 
        log_period=10,                       # 每经过 10 次训练迭代，就打印一次 log
        warmup_ratio=0.01,                   # 用于计算有多少 iteration 用于 warmup
        topk=(1,),
        dist=dict(                           # 1个数据/模型/流水并行组（即单 GPU 训练）
            data_parallel_size=1,          
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        evaluation=dict(                                 
            enabled=True,                    # 在训练期间启用 eval
        )
    )
)
```

+ LiBai 在 configs/common/train.py 中提供了训练配置的默认值，通过 `from configs.train import train` 导入 

+ 如果想要自定义一个可复用的训练配置，可以在一个单独的 train.py 中编写，然后使用时通过 `train = get_config("xxx/train.py").train` 导入。

+ 训练过程的总 iteration 数有 2 种设置方式：

    - 直接指定，例如 `train_iter=10000` 

    - 组合指定，`train_epoch = 500 ` + `iter_per_epoch = 10 ` ，总 iteration 数 =  `train_epoch * iter_per_epoch = 5000`

      若同时指定了 `train_iter`和`train_epoch` 2 个字段，在训练时将取两种方式的最大值（此处为 10000）。

configs/common/train.py 中各个配置项的含义如下：

```python
from libai.config import LazyCall
train = dict(
    
    # 保存输出文件的目录
    output_dir="./output",
    #  train_micro_batch_size: 每个 mini-batch 拆分到每个 GPU 上的 micro-batch 的大小
    #  train_mini_batch_size: 每个 GPU 在 1 个 step（iteration）中的训练样本数
    #  train_mini_batch_size = train_micro_batch_size * num_accumulation_steps。
    # global_batch_size = micro_batch_size  * num_accumulation_steps * data_parallel_groups
    train_micro_batch_size=32,
    global_batch_size=None,
    num_accumulation_steps=None,
    # training 的总 itreation 数
    train_iter=10000,
    # 实际总的 training iteration 次数将取 `train_iter` 和 `train_epoch * iter_per_epoch`两者的最大值
    train_epoch=0,  
    consumed_train_samples=0,
    consumed_valid_samples=0,
    train_samples=None,
    # warmup_ratio，指定 warmup 的 iteration 数
    warmup_ratio=0,  
    # 起始 iteration，一般不用手动设置。恢复训练时可被自动计算
    start_iter=0,
    # 是否在训练过程中开启 AMP (自动混合精度)
    amp=dict(enabled=False),
    # 是否启用 activation checkpointing
    activation_checkpoint=dict(enabled=False),  
    # NCCL 融合阈值（单位为 MB）。一般设为 0 来用与之前的 OneFlow 版本兼容
    nccl_fusion_threshold_mb=16,
    # NCCL 融合的最大操作数，一般设为 0 来与之前的 OneFlow 版本兼容
    nccl_fusion_max_ops=24,
    # 如果训练较大的模型，可以启用 ZeRO 优化。ZeRO 可以减少 optimize 阶段的内存消耗
    zero_optimization=dict(
        enabled=False,
        stage=1,
    ),
    
    # 保存模型的 checkpoints，period 用于指定保存模型的频率(单位为 iteration)，max_to_keep 指定最多保存多少个 checkpoints
    checkpointer=dict(period=5000, max_to_keep=100),  
    # 指定每个 batch 在单个 GPU 上用于 test 的样本数
    # 如果对数据并行组使用 8 个 GPU，并且 `test_micro_batch_size = 2`，则所有 GPU 每次迭代将总共使用 16 个样本。test 不进行梯度累积
    test_micro_batch_size=32,
    # 是否在训练期间进行评估，如果开启将在每经过`eval_period`次训练迭代后，执行一次评估
    # 可以设置最大 evaluation iteration，用于 validation/test
    # 可以使用自定义的 evaluator
    evaluation=dict(
        enabled=True,
        # evaluator for calculating top-k acc
        evaluator=LazyCall(ClsEvaluator)(topk=(1, 5)),  
        eval_period=5000,
        eval_iter=1e9,  # running steps for validation/test
        # Metrics to be used for best model checkpoint.
        eval_metric="Acc@1",
        eval_mode="max",
    ),
    # 指定要导入的模型 checkpoint 的路径
    load_weight="",
    # 日志输出的频率(单位为 iteration)
    log_period=20,
    # libai.scheduler 的相关参数，详见 libai/scheduler/lr_scheduler.py
    scheduler=LazyCall(WarmupCosineLR)(
        # In DefaultTrainer we will automatically set `max_iter`
        # and `warmup_iter` by the given train cfg.
        # 将根据给定的训练配置，设置 DefaultTrainer 的 max_iter 和 warmup_iter 值
        warmup_factor=0.001,
        alpha=0.01,
        warmup_method="linear",
    ),
    # 分布式相关的参数，详见 [Distributed_Configuration](https://libai.readthedocs.io/en/latest/tutorials/basics/Distributed_Configuration.html)
    dist=dict(
        data_parallel_size=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    ),
    
    # 设置随机种子值（必须是正数）
    seed=1234,
)
```