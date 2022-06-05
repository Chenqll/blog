https://www.csdn.net/tags/Mtzakg5sMjM3MjUtYmxvZwO0O0OO0O0O.html

什么是docker：docker是虚拟化容器技术，有三个主要概念：镜像（类）、容器（对象）、仓库。docker就是类似VM虚拟机一样的虚拟技术，体积小，运行速度快。

docker作用：可以把代码和环境一起打包部署到生产环境中。比如：我们写好的代码迁移到别的地方运行，不需要重新配置环境就能直接运行

docker底层：是用Go语言编写的

docker的三大特征：镜像、容器、仓库
镜像（类似于一个类）：包含了各种环境或者服务(tomcat)一个模板

容器（对象）：是镜像(run)起来之后的一个实例，可以把容器看做是一个简易版的Linux环境容器就是集装箱(logo上的集装箱)

仓库：是存放镜像的场所，最大的公开库是Docker Hub（https://hub.docker.com/）