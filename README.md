# Doctor for Cassava


## Update log
2020.12.12 <br>
> 1. 测试脚本，排查 infer Acc 过低的问题;
> 2. 混淆矩阵在 tensorboard 的可视化。


<br><br>
2020.12.13 <br>
> 1. fix Acc过低的问题，config 设置 multi-gpus=True, 保证多卡和单卡兼容;
> 2. opencv 代替 pil 进行数据增强，兼容 albumentaion 库;
> 3. 将标签文件的配置移入config;
> 4. 加入混淆矩阵在 tensotboard 显示。

<br><br>
2020.12.14 <br>
> 1. 权重保存时，同时保留只有 weight 的权重，方便 scp 传输;
> 2. 获取Cassava数据集的均值和方差.

<br><br>
2020.12.15 <br>
> 1. 增加 LR Range Test;
> 2. 联调 Git 端代码。

<br><br>
2020.12.16 <br>
> 1. 全部改为相对路径导入。

<br><br>
2020.12.17 <br>
> 1. 增加 gallery 和 query 的数据配置和处理;
> 2. 添加检索代码。

<br><br>
2020.12.19 <br>
> 1. 修正 model/build.py config.copy 的bug;
> 2. 检索抽特征代码;
> 3. 引入 model EMA，torch.seed。

<br><br>
2020.12.20 <br>
> 1. 加入 faiss 检索服务。

<br><br>
2020.12.22 <br>
> 1. 加入 imbalance 均衡采样;
> 2. 加入 FocalLoss 损失函数。