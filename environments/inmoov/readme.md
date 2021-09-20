# 看什么呢，还没见过个readme？

---------------------------------------------
A simple introduction of the code structre
---------------------------------------------------
`inmoov.py`包含了机器人的一些属性特征，其中里面有control的功能，也包含一个debugger模式

`immoov_p2p.py`是一个包含番茄和机器人的环境，他继承了SRLGym的结构，使用了`inmoov.py`中的机器人定义和控制功能

`joints_registry`包含了可以控制的joints信息


To run some test code for the inmoov environement
----------------------------------------------------
- 请先退出到根目录，也就是`RL-InmoovRobot`目录
- 执行 ``python -m environments.inmoov.test_inmoov``
- 如果要debug `inmoov.py`，请使用test_inmoov函数， 如果是debug `immoov_p2p.py` 文件，请使用test_inmoov_gym函数。

对于inmoov debug模式下不能显示全部joints的情况，暂时还无法解决，我给pybullet发了一个[issue](https://github.com/bulletphysics/bullet3/issues/2519)。目前比较好的解决方案是在
`joints_registry`里面改一下，注释掉不想看到的joints。

To train your first model (bugs alert!)
--------------------------------
环境还没有完全干净，里面可能存在些许问题，但是目前的进度允许我们进行简单的训练。
同样，请先到达项目的根目录。

- launch a ``visdom`` server to monitor the training process by: 
```
python -m visdom.server
```

- launch the training process:
```
python -m rl_baselines.train --env InmoovGymEnv-v0 --srl-model ground_truth --algo ppo2 --log-dir logs/ --num-timesteps 2000000
```
