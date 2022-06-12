# build oneflow
## 友链
- [Tradictional method](https://github.com/Oneflow-Inc/OneTeam/blob/63083c576fd1f6c627832651fad1ab1813028105/tutorial/howto_build_oneflow.md)
- [MORE GENERAL METHOD](https://github.com/Oneflow-Inc/conda-env)
- 补充[远程下载 miniconda](https://blog.csdn.net/weixin_43264420/article/details/118179287)

## 15号机无build流程
   ```bash
   cd oneflow 
   conda activate oneflow-dev-gcc7-v2
   mkdir build && cd build
   cmake .. -C ../cmake/caches/cn/cuda.cmake \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCUDNN_ROOT_DIR=/usr/local/cudnn
   make -j4
   . sourch.sh
   python3 -m oneflow --doctor

   ```

## 云平台编译
- 使用 16core-32Gi-mlu270 资源配置
- 只用使用 CPU-ONLY 的方法编译，make 选项可参见 oneflow 源码 readme 文件
  run this to config for CPU-only:
  ```
  cmake .. -C ../cmake/caches/cn/cpu.cmake
  ```
## 常见问题
- 编译过程出错：需要删除build环境重新编译 `rm -rf build`
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
