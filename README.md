# PyCR
**Pytorch for Classification and Retrieval** 
![Pipeline](https://raw.githubusercontent.com/xiaochengsky/PyCR/master/Pipeline.png)
<br>


## Classification
### Data
- [x] openCV
- [x] PIL
- [x] Many Augmentaion [SnapMix, CutMix, Mixup, CutOut, and so on...]


### BackBone
(fork from [TIMM](https://github.com/rwightman/pytorch-image-models))
- [x] TIMM
<br>

### Layer
- [x] dropblock
- [ ] Non local
- [ ] DCN
- [x] RGA Module
<br>
  
### Aggregation
- [x] GAP
- [x] GMP
- [x] GAP + GMP
- [x] GeM(w/o Fix)
- [x] SoftPool
<br>
  
### Head
- [x] BNNeck Head
- [x] Identity Head
- [x] Reduction Head
<br>
  
### Loss
- [x] CE
- [x] FocalLoss
- [x] Norm Softmax
- [x] Triplet 
- [x] CosFace
- [x] ArcFace
- [x] SnapMix
- [x] EQLoss
- [ ] Circle

### Other
- [x] EMA
- [x] PyTorch Optimizer and Scheduler


## Retrieval
Refer to PyRetri and FastReID design



<div style='display: none'>
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
> 2. 加入 FocalLoss。

<br><br>
2020.12.23 <br>
> 1. 修改 lr_range_test 路径问题;
> 2. 修复 EMA 针对多卡的 bug;
> 3. 修复 retrieval 部分的路径问题;
> 4. 检索效果和分类效果对齐, ema 更占优势。


<br><br>
2020.12.25 <br>
> 1. 添加配置项, 修复 pipeline 在 kaggle 平台的提交问题(不兼容 pynvml 指令)
> 2. 加入多尺度训练, 在 配置中统一ResizePad到(800, 800), 训练过程中重新随机Resize。


<br><br>
2021.01.19 <br>
> 1. 大修改;
> 2. 添加 config 0.895 的 snapmix 的 baseline;
> 3. 读取图像从 openCV 修改为 PIL;
> 4. 添加 engine/tmp_trainer.py 用于临时修改非正式代码;
> 5. 修改 engine/trainer.py;
> 6. layer 添加 rga 模块;
> 7. 添加 losses/snapmixloss.py;
> 8. 添加 nets/snapmix 的 res50 和 rga 模块的 res50;
> 9. 添加 layers 的 config 配置;
> 10. 分离 fineturn 和其它部分的学习率变化策略;
> 11. 修改 train.py;
> 12. 添加 snapmix 的处理，修改 torch, numpy, random 的种子


<br><br>
2021.01.20 <br>
> 1. 修改检索服务的 bug.


<br><br>
2021.01.22 <br>
> 1. 增加 eqloss + snapmix 的 loss函数
> 2. 增加 efficient 的 backbone

 </div>
