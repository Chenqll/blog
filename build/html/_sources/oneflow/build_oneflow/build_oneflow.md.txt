# build oneflow
- https://github.com/Oneflow-Inc/OneTeam/blob/63083c576fd1f6c627832651fad1ab1813028105/tutorial/howto_build_oneflow.md
- 补充
[远程下载 miniconda](https://blog.csdn.net/weixin_43264420/article/details/118179287)
- 删除build环境重新编译
  rm -f *
- 在15，16号机器，编译需要在gcc环境下，将clang环境改为gcc    
```
conda env create -f=dev/clang10/environment-v2.yml--->conda env create -f=dev/gcc7/environment-v2.yml
```
- 删除前面的build目录 然后新建一个build 重新cmake

- 缺少libunwind包，需要conda安装，需要在conda环境激活之前install，找conda包的途径 https://anaconda.org/conda-forge/libunwind

- 得出结论在15号机器别用ninja，成功的参考issue https://github.com/Oneflow-Inc/conda-env 主要就是用make -j4 的命令代替ninja -j4

- 注意：编译成功的oneflow每次都需要先进入conda环境：然后检查是否有编译好的环境：
```
(base) [chenqiaoling@oneflow-15 ~]$ conda activate oneflow-dev-gcc7-v2
(oneflow-dev-gcc7-v2) [chenqiaoling@oneflow-15 ~]

source build/source.sh

(oneflow-dev-gcc7-v2) [chenqiaoling@oneflow-15 ~]$ python -m oneflow --doctor
path: ['/home/chenqiaoling/oneflow/python/oneflow']
version: 0.8.0+cu102.git.2083cc9
git_commit: 2083cc9
cmake_build_type: Release
rdma: False
mlir: False
```
