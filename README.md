# **PyCR**
**Py**torch for **C**lassification and **R**etrieval <br>
![Pipeline](https://raw.githubusercontent.com/xiaochengsky/PyCR/master/Pipeline.png)
<br>


## **Classification**
### **Data**
#### Dataset Load
- [x] openCV
- [x] PIL
#### Sampler
- [x] Imbalance
#### Augmentation
- [x] general Augmentaions and albumentations
- [x] CutOut/Mixup/CutMix/SnapMix
<br>

### **BackBone**
(fork from [TIMM](https://github.com/rwightman/pytorch-image-models))
- [x] TIMM
<br>

### **Layer**
- [x] dropblock
- [ ] Non local
- [ ] DCN
- [x] RGA Module
<br>
  
### **Aggregation**
- [x] GAP
- [x] GMP
- [x] GAP + GMP
- [x] GeM(w/o Fix)
- [x] SoftPool
<br>
  
### **Head**
- [x] BNNeck Head
- [x] Identity Head
- [x] Reduction Head
<br>
  
### **Loss**
- [x] CE
- [x] FocalLoss
- [x] Norm Softmax
- [x] Triplet 
- [x] CosFace
- [x] ArcFace
- [x] SnapMix
- [x] EqLoss
- [x] Circle
<br>

### **Other**
- [x] TensorBoard
- [x] EMA
- [x] Multi-Scale Training
- [x] w/o Multi-GPUs
- [ ] TTA
<br>
<br>
<br>


## **Retrieval**
### **Dimension Process**
- [x] Normalize
- [x] PCA


### **Feature Enhance**
- [x] DBA
- [x] QE(w/o weight)

### **Distance Metric**
- [x] Faiss
- [x] Euclidean
- [x] Cosine

### **Re-rank**
- [ ] K-reciprocal
- [ ] K-reciprocal++
<br>
  

## **References**
[1] [FastReID](https://github.com/JDAI-CV/fast-reid) <br>
[2] [PyRetri](https://github.com/PyRetri/PyRetri) <br>
[3] [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) <br>


## Step by Step
