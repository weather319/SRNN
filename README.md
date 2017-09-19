## 介绍
这是一个稀疏递归神经网络的代码运行实例
![网络结构](https://github.com/weather319/SRNN/blob/master/RC_network.png)


## 要求
* python 3
* numpy
* pandas
* matplotlib
* scipy
* 

## 结果

* RC_network.py 是一个预测周期性曲线的基础
* load_data.py 是加载excel并清洗数据的文件
* train.py 是对水质数据简单训练的实例，包含了一些数据预处理方式
* TH_train.py 包含了 SRNN的最终实现和bp神经网络的实现对比

###水质预测曲线结果如图所示

![氨氮与溶解氧](https://github.com/weather319/SRNN/blob/master/result_A.png)
![CODmn、总磷、总氮](https://github.com/weather319/SRNN/blob/master/result_B.png)