# 环境
ubuntu18.04 + pytorch+ ros-melodic+gazebo

# 环境配置：
ubuntu18.04安装跳过
国内：ROS换阿里源
sudo gedit /etc/apt/sources.list
全部删除然后粘贴

deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse

deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse

deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse

更新
sudo apt-get update
sudo apt-get upgrade

一键安装ROS-melodic
sudo apt-get install curl && curl http://fishros.com/tools/install/ros-melodic | bash

一行代码搞定rosdepc
curl http://fishros.com/tools/install/rosdepc | bash

下载安装anaconda：

https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.11-Linux-x86_64.sh


bash Anaconda3-2021.11-Linux-x86_64.sh


创建安装虚拟环境：


conda env create -f py2.yaml

conda activate py2

cd catkin_wp/src

git clone https://github.com/ROBOTIS-GIT/turtlebot3.git

git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git

git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git

cd ..

catkin_make

source ~/catkin_wp/devel/setup.bash

roslauch turtlebot3_gazebo turtlebot3_stage1.launch

# 然后可以使用vscore来进行使用强化学习的代码

## 需要改几个地方，主要是路径问题。可以使用Ctrl+f 寻找 home来修改路径。


git clone 这个包 -不用编译




环境搭建完成？？？！！！




N_STATES =  28  N_ACTIONS = 5

效果：

PPO

https://www.bilibili.com/video/BV1zq4y1g7aM/

DQN:

https://www.bilibili.com/video/BV1BP4y1G7b1/

启动仿真环境：
roslaunch turtlebot3_gazebo turtlebot3_stage_2.launch
在vscore启动DQN2.py

效果：环境1：150回合收敛
      环境2：200回合收敛
      环境3：未测试
      环境4：未测试
      其他环境：未测试
如果你有好的注意可以提出来，欢迎交流。QQ：2650326396
拥抱开源
