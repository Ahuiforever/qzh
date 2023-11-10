# Installation

> Here are the steps you can install it on your own computer.

> Skip if the preconditions have been met.

## 1. Install Anaconda

<https://www.anaconda.com/download>

[Here](https://zhuanlan.zhihu.com/p/459601766) is the installation tutorial.

## 2. Create virtual conda environment

Open Anaconda Prompt, type in the following codes.

In which the 'qzh' stands for the name of the environment, customize it if like.

```powershell
$powershell
conda create -n qzh python=3.8 -y
conda activate qzh
```

( Recommended ) Or you can simply clone my environment without the 3rd step like this:

```powershell
$powershell
cd your_download_path/qzh
conda env create -f requirements.yml
```

( Recommended ) Or you can reshow the environment like this:

```powershell
$powershell
cd your_download_path/qzh
conda create  --name your_env_name --file spec-list.txt
```

## 3. Install all the requirements(skip if chose **Recommended** codes)

You may encounter some network problems with the following codes. Check [this](https://zhuanlan.zhihu.com/p/87123943).

```powershell
$powershell
# if you have VPN
pip install -r requirements.txt
# or if not
pip install -r requirements.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
```

# Usage

## 1. Prepare the nnmodel python file

Extract the qzh zip pack to your folders, make sure there is no Chinese character within path.

Modify the following codes to your own absolute path in nnmodel.py. There should be **3** different datasets:

- **train data** for training
- **dev data** for cross validation
- **test data** for testing, which might be not necessary
>- **all data** for calculating the global mean and standard deviation

```python
# python, nnmodel.py
train_data = DataReader(r'D:\Work\qzh\train')
dev_data = DataReader(r'D:\Work\qzh\dev')
test_data = DataReader(r'D:\Work\qzh\test')
```

## 2. Prepare your dataset

These key word must be contained within the path or the name of the *.csv file:

- coated000 : f
- df00 : df
- 000fcom000 : re1, im1 | first complex numer
- 000scom000 : re2, im2 | second complex numer
- 000nm : lambda
- radius00 : R
- thinly / thickly : C | 1 / 0

Examples:

```plaintext
./train/thickly/195fcom079/155scom000/350nm/coated005.csv
or
./dev/thinly/404nm/coated020/155scom000/195fcom079/df18.csv
or
./test/thinly_coated60_df18_700nm_195fcom079_155scom000.csv
or
...
```

## 3. Train
### Step1: Before training
Copy all the data to the dir named "all". Change the following codes.
```python
train_data = DataReader(r'E:\Work\qzh\train') ->
train_data = DataReader(r'E:\Work\qzh\all')
```
Then break line 182 of the code
```python
    # __Calculate the mean and standard deviation
    # _mean = torch.mean(self.x, dim=0)
    # _var = torch.var(self.x, dim=0)
    # print(f"mean: {_mean}"
    #       f"var: {_var}")
    # __=========================================
```
```powershell
$powershell
conda activate qzh
cd ./qzh
python train.py
```
### Step2: 
```python

```
## 4. Reasoning

*Still writing...*