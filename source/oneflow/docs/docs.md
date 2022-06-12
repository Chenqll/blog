# docs
## build docs
- https://github.com/Oneflow-Inc/OneTeam/blob/master/tutorial/howto_write_api_docs.md
- https://github.com/Oneflow-Inc/community/pull/12

```
  git clone git@github.com:Oneflow-Inc/oneflow-api-cn.git
  cd oneflow-api-cn
  python3 -m pip install pybind11 --user // pybind11 是一个 header-only 的库，换句话说，只需要 C++ 项目里直接 include pybind11 的头文件就能使用
  python3 -m pip install -r docs/requirements.txt --user
  python3 setup.py install --user
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
  #### 生成表格 （以重构 tensor 模块为例）
   - tensor 模块对标 pytorch 文档的 tensor 模块，发现 *“pytorch 的 tensor 模块以数据类型和 device 类型划分关于tensor 的算子，并采用表格的形式展现”*
   - 使用 furo 自定义的[表格类型](https://pradyunsg.me/furo/reference/tables/)，在 tensor.rst 文件内构造表格形式，并使用 `:class` 指令，重定向到算子详情页。


  
