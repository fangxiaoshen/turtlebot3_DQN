# 环境
ubuntu18.04 + pytorch+ ros-melodic+gazebo

# 环境配置：
ubuntu18.04安装跳过


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
如果你有好的注意可以提出来，欢迎交流。

# QQ群：877273841

# QQ群所有相关学习资料

# 有相关问题也可以在QQ群中提出大家一起交流

# 拥抱开源

# 未来想在机器人方面进行研发教育可以联系本人微信：ffd2650 base ：广州番禺，可内推

# 关于我的：目前只是一个大四的学生，如果有什么不对的地方，请多多指正

# 深度强化学习-学习资源推荐：
## 微信公众号：
### 深度学习实验室
### RLCN
### OpenDILab

# 目前我在干什么：
## 把算法应用到实际小车上，本人很懒，所以还在拖（狗头）
# 未来思路：
## 可以使用改变传入的数据进行深度学习。
## 从深度强化学习算法上进行改进，可以应用最前沿的算法。
## 可以加入transiform 到强化学习和深度学习中
## 可以通过改变输出，就是直接进行电机控制
。。。。

# 有空我可以把一个虚拟机的训练镜像打包好分享出来
# 有空会在ROS2上迁移
# 有空会分享一下我的小车的搭建https://blog.csdn.net/qq2650326396/article/details/122161688?spm=1001.2014.3001.5502





















