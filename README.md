# AFFGCN架构  交通流量预测
Attention Feature Fusion base on spatial-temporal Graph Convolutional Network
https://github.com/guoshnBJTU/ASTGCN-r-pytorch
https://github.com/wanhuaiyu/ASTGCN
![图片1](https://user-images.githubusercontent.com/49970610/230247050-ed36f004-e208-4888-9443-48b32ac3117b.jpg)



# 交通数据集

|  区域   |    时间区间     |    检测器    |      数值特征      |
| :-----: | :-------------: | :----------: | :----------------: |
| PEMS-08 | 2018.01-2018.02 | 307 个检测器 | 流量，占有率，速度 |
| PEMS-04 | 2016.07-2016.08 | 170 个检测器 | 流量，占有率，速度 |

# 天气数据集

|     区域     |    时间区间     | onehot 特征 | 数值特征 |
| :----------: | :-------------: | :---------: | :------: |
|  旧金山湾区  | 2018.01-2018.02 |    25 维    |  13 维   |
| 圣贝纳迪诺市 | 2016.07-2016.08 |    16 维    |  11 维   |

# 数据预处理
weather为天气数据的预处理，主要包括天气数据的原始数据文件处理，对天气数据的时间切片调整至与交通数据相同维度

# onehot

**skyc1、skyc2**                                                                                                                                       **wxcodes**

| 特征代码 | One-hot向量           | &emsp;&emsp;&emsp;&emsp; | 特征代码 | One-hot向量             |
| -------- | --------------------- | ------------------------ | -------- | ----------------------- |
| CLR      | [  1 , 0, 0, 0, 0, 0] | &emsp;&emsp;&emsp;&emsp; | HZ       | [  1, 0, 0, 0, 0, 0, 0] |
| FEW      | [  0, 1, 0, 0, 0, 0]  | &emsp;&emsp;&emsp;&emsp; | RA       | [  0, 1, 0, 0, 0, 0, 0] |
| VV       | [  0, 0, 1, 0, 0, 0]  | &emsp;&emsp;&emsp;&emsp; | BR       | [  0, 0, 1, 0, 0, 0, 0] |
| SCT      | [  0, 0, 0, 1, 0, 0]  | &emsp;&emsp;&emsp;&emsp; | RA BR    | [  0, 0, 0, 1, 0, 0, 0] |
| BKN      | [  0, 0, 0, 0, 1, 0]  | &emsp;&emsp;&emsp;&emsp; | BR       | [  0, 0, 0, 0, 1, 0, 0] |
| OVC      | [  0, 0, 0, 0, 0, 1]  | &emsp;&emsp;&emsp;&emsp; | BCFG     | [  0, 0, 0, 0, 0, 1, 0] |
|          |                       | &emsp;&emsp;&emsp;&emsp; | BCFG     | [  0, 0, 0, 0, 0, 0, 1] |