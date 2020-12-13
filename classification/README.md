# 用于分类任务

# Update log
2020.12.12 <br>
1. 测试脚本，排查 infer Acc 过低的问题:
   <add> tools.tester.py
   
2. 混淆矩阵在 tensorboard 的可视化:
   <modify> engine.trainer
   

2020.12.13 <br>
1. fix Acc过低的问题，config 设置 multi-gpus=True, 保证多卡和单卡兼容
2. opencv 代替 pil 进行数据增强，兼容 albumentaion 库

