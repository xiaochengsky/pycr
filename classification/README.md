# 用于分类任务

## Update log
2020.12.12 <br>
1. 测试脚本，排查 infer Acc 过低的问题:
> 1. add-> tools.tester.py
    
2. 混淆矩阵在 tensorboard 的可视化:
> 1. modify-> engine.trainer

<br><br>
2020.12.13 <br>
> 1. fix Acc过低的问题，config 设置 multi-gpus=True, 保证多卡和单卡兼容
> 2. opencv 代替 pil 进行数据增强，兼容 albumentaion 库
> 3. 将标签文件的配置移入config
> 4. 加入混淆矩阵在 tensotboard 显示 

<br><br>
2020.12.14 <br>
> 1. 权重保存时，同时保留只有 weight 的权重，方便 scp 传输
> 2. 获取Cassava数据集的均值和方差

   