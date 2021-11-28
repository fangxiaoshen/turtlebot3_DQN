# 环境
ubuntu18.04 + pytorch+ ros-melodic+gazebo

环境配置：
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
conda env export > py27.yaml

环境搭建完成？？？




N_STATES =  28  N_ACTIONS = 5

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
