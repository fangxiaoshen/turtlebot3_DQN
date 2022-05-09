# 环境

## ubuntu18.04 + pytorch+ ros-melodic+gazebo11

# 环境配置：

ubuntu18.04安装跳过 ，虚拟机和双系统都可以

虚拟机：[在虚拟机中安装Ubuntu 18.04 - 简书 (jianshu.com)](https://www.jianshu.com/p/c743aaa847de)

双系统：[(13条消息) Windows 10 安装ubuntu 18.04 双系统（超详细教程）_Ycitus的博客-CSDN博客_win10安装ubuntu双系统](https://blog.csdn.net/qq_43106321/article/details/105361644)

## ROS-melodic 安装：

```shell
wget http://fishros.com/install -O fishros && . fishros
```

## rosdep：

```
wget http://fishros.com/install -O fishros && . fishros
```

## 下载安装anaconda：

https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.11-Linux-x86_64.sh


bash Anaconda3-2021.11-Linux-x86_64.sh

## 创建安装虚拟环境：

```shell
git clone https://github.com/Crawford-fang/turtlebot3_DQN.git
cd turtlebot3_DQN
conda env create -f py2.yaml
```

## 创建工作空间：

```shell
mkdir -p ws/src
cd ws/src
git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
cd ..
```

## 修改激光雷达线数：

参考：[TurtleBot3 (robotis.com)](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#machine-learning)

```
roscd turtlebot3_description/urdf/
gedit turtlebot3_burger.gazebo.xacro
#如果想可视化激光雷达，把下面改成true
<xacro:arg name="laser_visual" default="false"/> 
#把激光雷达数据改成24
<scan>
  <horizontal>
    <samples>24</samples> # The number of sample. Modify it to 24
    <resolution>1</resolution>
    <min_angle>0.0</min_angle>
    <max_angle>6.28319</max_angle>
  </horizontal>
</scan>

```

![image-20220509170306732](C:\Users\26503\AppData\Roaming\Typora\typora-user-images\image-20220509170306732.png)

## 在工作空间下运行，安装ROS功能包全部依赖：

```shell
cd ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

## 然后可以使用vscore来进行使用强化学习的代码

## 需要改几个地方，主要是路径问题。可以使用Ctrl+f 寻找 home来修改路径。


N_STATES =  28  N_ACTIONS = 5

效果：

PPO：

https://www.bilibili.com/video/BV1zq4y1g7aM/

DQN:

https://www.bilibili.com/video/BV1BP4y1G7b1/

启动仿真环境：
roslaunch turtlebot3_gazebo turtlebot3_stage_2.launch
在vscore启动DQN2.py

# 交流QQ群：877273841



# 深度强化学习-学习资源推荐：

## 微信公众号：

### 深度学习实验室

### RLCN

### OpenDILab

# 未来思路：

## 可以使用改变传入的数据进行深度学习。

## 从深度强化学习算法上进行改进，可以应用最前沿的算法。

## 可以加入transiform 到强化学习和深度学习中

## 可以通过改变输出，就是直接进行电机控制

# 有空我可以把一个虚拟机的训练镜像打包好分享出来

# 有空会在ROS2上迁移

# 有空会分享一下我的小车的搭建https://blog.csdn.net/qq2650326396/article/details/122161688?spm=1001.2014.3001.5502

# 常见问题及解决方法

## 问题-路径问题

代码使用的是绝对路径，修改一下路径即可。

## 加载模型：

加载训练模型开关;true为加载训练好的模型，false是不加载模型进行训练；

self.loal_model=false




