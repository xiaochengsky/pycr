## **Prerequisites**
* Python 3.6+
* PyTorch 1.6+
* CUDA (corresonding to PyTorch)

## **Installation**
### **Prepare environment**
1. Create a conda virtual environment and activate it.
```
conda create -n pycr python=3.7 -y
conda activate pycr
```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
Note: Make sure that your compilation CUDA version and runtime CUDA version match. 
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

### **Install PyCR**
Clone the repository and then install it:
```
git clone https://github.com/xiaochengsky/PyCR.git
cd PyCR
pip install -r docs/requirements.txt
```

### **Verification**
TODO






