# docs
## build docs
- https://github.com/Oneflow-Inc/OneTeam/blob/master/tutorial/howto_write_api_docs.md
- https://github.com/Oneflow-Inc/community/pull/12

```
  git clone git@github.com:Oneflow-Inc/oneflow.git
  cd oneflow
  python3 -m pip install pybind11 --user // pybind11 是一个 header-only 的库，换句话说，只需要 C++ 项目里直接 include pybind11 的头文件就能使用
  python3 -m pip install -r docs/requirements.txt --user
  python3 setup.py install --user
```
- 如果修改了 `oneflow` 目录下的内容，比如 `oneflow/python/oneflow/framwork/xxx.py` 内的 docstring ，想要看到效果，需要进入 `oneflow/build` 目录重新编译 oneflow 后再编译 html
  ```bash
  cd ../build && make -j4
  cd ../docs && make html
  ```

## rebuild docs
### **Sphinx** 

  - use sphinx
    https://zhuanlan.zhihu.com/p/27544821
  
  Sphinx 是一个基于 Python 的文档生成项目。使用 reStructuredText 格式

  英文文档：https://www.sphinx-doc.org/en/master/

  Sphinx 生成的文档架构如下：

  - build 生成的文档 html 文件
  - source/conf.py 包含 Sphinx 项目配置的 Python 脚本，和一些额外的配置
  - source/index.rst 项目的根文档

  Sphinx `Extension` 使用 sphinx 内置扩展配置，自定义 Sphinx 的“骨架” 表现形式，使用方法如下：
  - 在 source/conf.py 的 extensions 下加入对应的扩展名，下载对应扩展包，即可使用扩展。
  - 可以自定义扩展的属性

      ```python
      # 以添加 autosummary 扩展为例：
      extensions = [
      'sphinx.ext.autosummary ',
      ] 
      ...
      ...
      # build the templated autosummary files
      autosummary_generate = True
      numpydoc_show_class_members = False
      ...
      ```

  - 更多[扩展](https://www.sphinx-doc.org/en/master/usage/extensions/index.html)

### **RST** 
  
  reStructuredText 是一种轻量级标记语言。与 Markdown 非常相似。

  ##### RST 指令
  - 以下是文档目录树结构的描述，`.. toctree::` 声明了一个树状结构（toc 即 Table of Content），`:maxdepth: 1` 表示目录的级数（页面最多显示一级），`:caption: Contents: ` 用于指定标题文本，将会生成下图目录结构 **图**
  ```rst
  # 以 source/index.rst 为例
  .. toctree::
    :maxdepth: 1
    :caption: OneFlow Python API

    oneflow
    nn
    functional
    tensor
    ...
  ```

### **Furo** 
  - sphinx 文档生成的一种样式，可以实现主题的整体外观和感觉的定制，[文档](https://pradyunsg.me/furo/)

## 按照模块进行重构
interface is consistent with pytorch 的接口顺序我都改了
### oneflow
#### pytorch 有的然而 oneflow 没有的接口,包括在oneflow.tensor下而没在oneflow下的，比如 `oneflow.Tensor.triu`
- oneflow/tensor 下
  
is_storage is_complex is_conj get_default_dtype set_default_tensor_type set_flush_denormal

- oneflow/Tensor/creation op 下
    sparse_coo_tensor 
    asarray 
    range 
    logspace 
    empty_like 
    empty_strided    
    full_like
    quantize_per_tensor
    quantize_per_channel
    dequantize
    complex
    polar
    heaviside
- oneflow/tensor/indexing...下
  adjoint
  conj
  dsplit
  column_stack
  dstack
  hstack
  index_add
  moveaxis
  row_stack
  diagonal_scatter
  select_scatter
  slice_scatter
  scatter_reduce
  take
  take_along_dim
  tensor_split
  vsplit
  vstack

- Generator

- oneflow/random sampling/In-place random sampling 除了normal_ 整个目录没有 ??
- oneflow/random sampling xia
  multinomial
  poisson
  rand_like
  randint_like
  randn_like
- Quasi-random sampling 无
- oneflow/pall xia
  get_num_threads
  get_num_interop_threads
  set_num_interop_threads
- oneflow/math op/pointwise op
  absolute
  angle
  arctan2
  bitwise_and
  bitwise_not
  bitwise_or
  bitwise_xor
  bitwise_left_shift
  bitwise_right_shift
  conj_physical
  copysign
  deg2rad
  divide
  digamma
  exp2
  fake_quantize_per_channel_affine
  fake_quantize_per_tensor_affine
  fix
  float_power
  floor_divide
  frac
  frexp
  gradient
  imag
  ldexp
  lerp
  lgamma
  logaddexp
  logaddexp2
  logit
  hypot
  i0
  igamma
  igammac
  multiply
  mvlgamma
  nan_to_num
  nextafter
  polygamma
  positive
  quantized_batch_norm
  quantized_max_pool1d
  quantized_max_pool2d
  rad2deg
  real
  remainder
  signbit
  sinc
  tanh
  true_divide
  trunc
  xlogy
  subtract
- oneflow/math op/reduce op
  aminmax
  all
  dist
  logsumexp
  nanmean
  nanmedian
  mode
  norm
  nansum
  quantile
  nanquantile
  std_mean
  var_mean
  count_nonzero
  unique
  unique_consecutive
- oneflow/math op/compare op
  allclose
  ge
  greater_equal
  greater
  isclose
  isfinite
  isin
  isposinf
  isneginf
  isreal
  kthvalue
  less_equal
  less
  maximum
  minimum
  fmax
  fmin
  not_equal
  msort
- oneflow/math op/spectal op 所有都没有
- oneflow/math op/other op
  atleast_1d
  atleast_2d
  atleast_3d
  bincount
  block_diag
  broadcast_tensors
  broadcast_to
  broadcast_shapes
  bucketize
  cartesian_prod
  cdist
  combinations
  corrcoef
  cov
  cross
  cummax
  cummin
  diag_embed
  diagflat
  diff

  fliplr
  flipud
  kron
  rot90
  gcd
  histc
  histogram
  histogramdd
  lcm
  logcumsumexp
  ravel
  renorm
  repeat_interleave
  trace
  tril_indices
  triu
  triu_indices
  vander
  view_as_real
  view_as_complex
  resolve_conj
  resolve_neg
- oneflow/math op /BLAS and LAPACK Operations 下
  addbmm
  addmv
  addr
  baddbmm
  chain_matmul
  cholesky
  cholesky_inverse
  cholesky_solve
  eig
  geqrf
  ger
  inner
  inverse
  det
  logdet
  slogdet
  lstsq
  lu
  lu_solve
  lu_unpack
  matrix_power
  matrix_rank
  matrix_exp
  mv
  orgqr
  ormqr
  outer
  pinverse
  qr
  svd
  svd_lowrank
  pca_lowrank
  symeig
  lobpcg
  trapz
  trapezoid
  cumulative_trapezoid
  triangular_solve
  vdot
- oneflow/utilities 下的都没有
### OneFlow.nn 
#### content 目录缺少 Recurrent Layers/Transformer Layers/Shuffle Layers/Utilities/Quantized Functions/Lazy Modules Initialization
- oneflow.nn/recurrent layers 下    
    nn.RNNBase
    nn.RNN
    nn.LSTM
    nn.GRU
    nn.RNNCell
    nn.LSTMCell
    nn.GRUCell
- oneflow/Transformer Layers 下都没有
- oneflow.nn/container/Global Hooks For Module 都没有？？
- oneflow.nn/Convolution Layers 下的 lazy 初始化的类都没有？
    nn.LazyConv1d
    nn.LazyConv2d
    nn.LazyConv3d
    nn.LazyConvTranspose1d
    nn.LazyConvTranspose2d
    nn.LazyConvTranspose3d
    nn.Unfold
    nn.Fold
- oneflow.nn/pooling layer

### oneflow.lingle
- oneflow.linalg/Matrix Properties 下 
  diagonal 是oneflow.diagonal 而不是 oneflow.lingle.diagonal
  det
  slogdet
  cond
  matrix_rank
- oneflow.linalg/Decompositions 下都没有
- oneflow.linalg/Inverses 下都没有
- oneflow.linalg/Matrix Functions下都没有
- oneflow.linalg/Matrix Products
  matmul 是 oneflow.matmul 和 oneflow.Tensor.matmul
  cross
  multi_dot
  householder_product
- oneflow.linalg/Tensor Operations 下都没有
- oneflow.linalg/misc
- oneflow.linalg/Experimental Functions

### oneflow.nn.init
- oneflow.nn.init/
### oneflow.optim
- oneflow.optim/base 下
  Optimizer.add_param_group 无文档
- oneflow.optim/al 下
  Adadelta
  Rprop
  NAdam
  SparseAdam
  Adamax
  ASGD
  LBFGS
  RAdam
- oneflow.optim/lr_scheduler
  MultiplicativeLR
  CyclicLR
  OneCycleLR
- oneflow.optim/Stochastic Weight Averaging 都没有

#### pytorch 重构方法
- 主要讲的是 pytorch 的数据 Tensor 和常见操作等
- Tensor-> 张量的内置属性/Tensor/creation op -> 创建操作

  
## 解惑
#### /oneflow-api-cn/docs/source/cn/activation_ops.py 里的中文是哪来的？
    首先搞懂英文的是哪来的，是动态设置docstring来的，C++的接口需要添加docstring，于是在 'oneflow/python/oneflow/framework/docstr'下设置.py文件对由C++导出的接口设置docstring  https://github.com/Oneflow-Inc/OneTeam/blob/master/tutorial/howto_write_api_docs.md

    通过调用 docreset.reset_docstr，把原有的 __doc__ 替换为中文翻译。__doc__ 是python对象的一个属性
  



#### /oneflow-api-cn/docs/source/autograd.rst 这个rst文件是什么意思，什么作用
    
     make html 命令，其实就是利用了工具 sphinx-build，将 oneflow 中对象的 docstring 提取出来，生成 HTML 文件。

     make html 时的内部原理如下，即 sphinx 先读取 conf.py，并在其中 import oneflow，然后读取 *.rst 文件，确认要生成哪些 Python 接口的文档，然后提取它们的 docstring，并生成 HTML 文件。



#### 出问题有可能是oneflow仓库没有更新
      到这里 https://docs.oneflow.org/master/index.html 下载安装最新的oneflow：
      python3 -m pip install -f https://staging.oneflow.info/branch/master/cu112 --pre oneflow
      出现没有CUDA的问题 也有可能是安装了oneflow的cpu版本
  
#### 如果修改了docreset的init文件
      pip uninstall docreset
      python setup.py install 重新安装
  
#### doc里的代码test过不了可能是给出的输出和源输出不一致，此时
      ipython 进入python环境
      把 docstring里的打码复制进去，看看是哪里报错

#### 使用oneflow-api-cn里面的guide的readme出现的问题
##### sphinx-build not found
      原因是他不在环境变量里
      首先找到sphinx-build在哪里，发现在~/.local/bin
      ```
      查看一下：
      ~/.local/bin/sphinx-build 
      ```
      于是把它加在环境变量里
      ```bash
      export PATH=~/.local/bin:$PATH 
      ```
##### 找不到oneflow 
      ```bash
      (base) [chenqiaoling@oneflow-15 docs]$ make html
      `Running Sphinx v3.5.4

      Configuration error:
      There is a programmable error in your configuration file:

      Traceback (most recent call last):
      ...
      ...
      ModuleNotFoundError: No module named 'oneflow._oneflow_internal'

      make: *** [html] Error 2`
      ```
      此时打印一下
      ```
      Successfully installed oneflow-0.8.0.dev20220510+cu102
      (base) [chenqiaoling@oneflow-15 ~]$ python -m oneflow --doctor
      Traceback (most recent call last):
        File "/home/chenqiaoling/miniconda3/lib/python3.9/runpy.py", line 188, in _run_module_as_main
          mod_name, mod_spec, code = _get_module_details(mod_name, _Error)
        File "/home/chenqiaoling/miniconda3/lib/python3.9/runpy.py", line 147, in _get_module_details
          return _get_module_details(pkg_main_name, error)
        File "/home/chenqiaoling/miniconda3/lib/python3.9/runpy.py", line 111, in _get_module_details
          __import__(pkg_name)
        File "/home/chenqiaoling/oneflow/python/oneflow/__init__.py", line 21, in <module>
          import oneflow._oneflow_internal
      ModuleNotFoundError: No module named 'oneflow._oneflow_internal'
      ```
      发现她是在oneflow包里找的，在的 PYTHONPATH 里，导入的第一优先级，是导入你的这个源码里的 oneflow，但是你的源码 oneflow 没有编译通过，所以就凉了。
        
      把环境变量删除 ```unset PYTHONPATH```,把这个环境变量删除。就不会第一导入你的源码路径。这样就可以顺位找其它的 oneflow了（pip安装的）
#### Python 版本不对
     想要修改 .py 文件的 docstring ，必须重新编译 oneflow，此时要求内置调用的 oneflow 是编译安装的而不是 pip 安装的。
     编译好的 oneflow 需要保证内置的 python 版本与使用 sphinx-build 时的 python 版本一致。当一开始就是在 conda 环境的话，不同功能使用时 API 版本不对的问题就不会出现。

####  

#### 关于signature
      如果不加 signature，又是 C++ 直接导出的接口，那么 signature 其实是无参的。就可能是错的。

      如果是 python 接口，不加 signature 是没问题的，会提取正确。

#### 为了避免 conflicts
      需要在 master 分支上先 ```git pull origin master``` 再 checkout -b

#### 关于 oneflow.nn 和 oneflow.nn.functional 重复的一些问题：
- oneflow.nn.functional 是从原生的 C++ 框架内导出的接口
- oneflow.nn 更多是在 python 的范围内拼凑的
- 哪一个更常用？两者之间如何转化写出nn有而nn.functional 没有的 nn.functional 的接口？



  
## 如何重构 DocsV 0.8.0

### 前情提要：为什么要进行文档重构

- 现有文档：太简单，仅仅是一个简单的罗列，实现的功能也只是检索。
  - [快速启动现有文档](https://github.com/Oneflow-Inc/OneTeam/blob/master/tutorial/howto_write_api_docs.md)
  
- 目标文档：对标 pytorch，将 API 按照功能分类，实现多种表现形式（表格，summary 等）
  - 已完成的[目标文档](https://github.com/Oneflow-Inc/oneflow/pull/8328)重构



### 快速上手重构
   确定要重构的版块，目前已重构 oneflow、nn、nn.functional、oneflow.tensor 四个板块。
   将已有文档与 pytorch 文档作对比（包括显示对比与源码对比），确定重构的目标。大致有以下四种目标
   - 生成 summary
   - 生成表格

  #### 生成 summary（以重构 oneflow 模块为例）：
   - oneflow 模块对标 pytorch 文档的 torch 模块，发现 *“torch 模块实现了以功能划分算子，并对每一个算子有 summary，点击后可以进入算子详情页”*
   - 通过阅读 pytorch 源码和 Sphinx 文档发现，可以使用 sphinx 的 `autosummary` [扩展](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html) 实现上述 torch 模块的功能。
     - 首先在 `source/conf.py` 中引入 `autosummary` 的扩展，并设置扩展属性。
     - 然后在 `oneflow.rst` 实现实例化 summary，以下是在 `source/oneflow.rst` 内使用 `autosummary` 扩展的描述：
       `tensor` 加横线声明了一个小标题，`.. autosummary::` 插入一个表格，其中包含指向文档项目的链接，以及每个项目的简短摘要简介（文档字符串的第一句）,`toctree`向 `sphinx-autogen` 脚本发出信号，表明应该为该指令中列出的条目生成存根页面, `nosignatures` 表示不在在列表中显示函数签名。
        ```rst
        Tensor
        -------------------------------------------

        .. autosummary::
            :toctree: generated
            :nosignatures:

            is_tensor
            is_floating_point
            is_nonzero
            numel
            set_printoptions
        ```
        ##### autosummary TIPS
        - autoaummary 是 sphinx 的 extension，应该具体查看 sphinx 的 extension 文档
  #### 交叉引用：
  ##### 引用文档
  - 首先每一个 rst 文件都当做一个文档，并且都需要在 source/index.rst 下的 toctree 进行注册
  - 当在一个文档内想要引用另一个文档时，使用 `:doc:` 指令，例如 `:doc:example1` 引用 example1 文档
  ##### 引用小标题
  - 在一个小标题上添加指令 `.. _example2:`，然后在另一个文件使用 `:ref:`example2`` 指令引用该标题。
  #### 生成表格 （以重构 tensor 模块为例）
   - tensor 模块对标 pytorch 文档的 tensor 模块，发现 *“pytorch 的 tensor 模块以数据类型和 device 类型划分关于tensor 的算子，并采用表格的形式展现”*
   - 使用 furo 自定义的[表格类型](https://pradyunsg.me/furo/reference/tables/)，在 tensor.rst 文件内构造表格形式，并使用 `:class` 指令，重定向到算子详情页。

### 报错解决
- undefined label: random-sampling

一般都是[交叉引用](#交叉引用)时出现的问题

- docstring 中写的语法报错

<!-- 
## torch 和 oneflow distributed 的对比
### torch 构建 distributed 模块的方式
- 有 overview，oneflow 也有对应的文档可供参考


## 共有模块
- oneflow.comm == torch.distributed 的 collective function

### oneflow 的distributed 模块的特点：
#### general
- 多设备训练会涉及到设备间沟通的成本
- **数据并行** 会使得反向传播时，各个设备更新的计算的模型梯度不同，导致设备间的模型不同。此时需要通过对各个设备的梯度执行 ALLREDUCE 策略，保证模型一致。适合数据集大，模型小的模型（resnet 50）
- **模型并行** 会省去 ALLREDUCE，但是需要 Broadcast。适合模型过大，一个设备放不下的场景（bert）
- **流水并行** 当模型过大时，可以将神经网络切分为多个阶段，分发到不同设备上。
- **混合并行** 多种策略混用。
#### sbp
- 集群的全局视角


### oneflow.distributed
.. note ::
    Please refer to `PyTorch Distributed Overview <https://docs.oneflow.org/master/parallelism/01_introduction.html>`__
    for a brief introduction to all features related to distributed training.

- backens **unkonw**
- basic **unkonw**
  - 写 global view 的基础模块
  - 加上一句话，与 pytorch 内容重合的部分
  - 是否要加 sbp signature
- initialize **unkon**
  - 我们需要初始化么？
  - 可以改为 创建 global tensor 么
- after initialize **konw in oneflow.env**
- Distributed Key-Value Store **unkonw**
- group **unkonw**
- Point-to-point communication **konw in oneflow.comm of send and recv**
- collective function **konw in oneflow.comm of remain**
- Multi-GPU collective functions **konw in oneflow docs and need reconstruct**


## 重构思路
- 1. 对标 pytorch 的 torch.distributed 先引到 docs 文档
- 2. 直接介绍特性的 global view 和 三个重要概念
- 3. basic 模块介绍三个重要概念，引出接口
- 4. 对标 pytorch 的初始化模块，创造 create global tensor 模块
- 5. Post-initialization 引入 oneflow.env 接口
- 6. communication collective 引入 oneflow.comm 接口
- 7. launching
## 疑问
- 1. 是否可以引入 oneflow.env 和 oneflow.comm 到 oneflow.distributed 目录下？
- 2. 文字部分是以一句话简单介绍为好，还是具体解释？
- 3. 与 pytorch 重合的部分 nn.parallel.DistributedDataParallel 不在此部分做介绍可否？


## 7.11-7.16 的合并工作
- 1. ChenGuoLiang & LiuXuan 负责基础模块的 review
  
    主要方式是:
      按照chengguoliang 的 autograd 的沟通方式，找到对应全职同事将 pytorch 和 oneflow 的文档都过一遍（分模块，分接口），之后按照建议修改

- 5. 文档工作形成文档
- 7. oneflow.nn.init
- 8. 环境变量协调



review for ready
|  task   | 负责人  |
|  ----  | ----  |
| oneflow  | Chengguoliang |
| oneflow.nn  | Chengguoliang |
|oneflow.Tensor|Chengguoliang|
|oneflow.linalg|Chengguoliang|
|oneflow.nn.init|Chengguoliang|
|oneflow.utils.data|Chengguoliang|
|oneflow.optim|Chengguoliang|
|oneflow.autograd|Chengguoliang|
|tensor attribute|Liuxuan|
|oneflow.cuda|Liuxuan|
|oneflow.nn.functional|Liuxuan|
|oneflow.rnn 报错问题|Liuxuan|
|conflict 问题|Liuxuan|
|oneflow.nn.graph|ChenQiaoLing(Xv Xiaoyu)|
|oneflow.distributed|ChenQiaoLing(Chen HouJiang,Han BinBin)|
|oneflow.oneEmbedding|ChenQiaoLing(Guoran)|
|环境变量|ChenQiaoLing()|


## 环境变量翻译工作
https://github.com/Oneflow-Inc/OneTeam/issues/654
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

#####################
Environment Variables
#####################

NCCL has an extensive set of environment variables to tune for specific usage.

ONEFLOW_COMM_NET_IB_HCA
--------------------------------

当服务器存在多张IB网卡(可通过ibstatus查看)时，系统默认使用第一张IB网卡进行comm_net通信，当设置了这个环境变量后，系统会遍历所有的IB网卡，找到对应名字的网卡

Values accepted
^^^^^^^^^^^^^^^
默认为空，形如：mlx5_0:1、mlx5_1:1，当端口为0的时候，默认为1，表示使用第一个端口。


## distributed 
总的来说 分为三个
1. 共有的命名为 basic 放上 postinit 下的接口
2. local 有的 放在ddp 目录下，放上 community collection 下的接口
3. global 特有的 只有两个接口 to_global to_local


删除 global view placement sbp三个 改为 basic ddp global tensor

global tensor 需要加示例代码

ddp 的集合通信内容需要仿照 pytorch 写一些关于 all_reduce 的描述

ddp的 docstring 加一些内容 -->