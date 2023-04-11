# AFFGCN-Architecture  （Traffic Flow Prediction）
Attention Feature Fusion base on spatial-temporal Graph Convolutional Network（AFFGCN）
![模型总体架构](https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/%E6%A8%A1%E5%9E%8B%E6%80%BB%E4%BD%93%E6%9E%B6%E6%9E%84.jpg)

# Reference

@inproceedings{guo2019attention,
  title={Attention based spatial-temporal graph convolutional networks for traffic flow forecasting},
  author={Yangan，Wangwei，Liucheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={922--929},
  year={2022}
}

# Datasets

Step 1: Download PEMS04 and PEMS08 datasets provided by [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data)

Step 2: Process dataset

# Traffic dataset

![traffic dataset](https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/111.jpg)

| district |  Time interval  | detector |       feature       |
| :------: | :-------------: | :------: | :-----------------: |
| PEMS-08  | 2018.01-2018.02 |   307    | flow, occupy, speed |
| PEMS-04  | 2016.07-2016.08 |   170    | flow, occupy, speed |

# Weather Dataset

![weather dataset](https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/21.jpg)

|        district        |  Time interval  | one-hot | Numeric Features |
| :--------------------: | :-------------: | :-----: | :--------------: |
| San Francisco Bay Area | 2018.01-2018.02 |   25    |        13        |
|  San Bernardino city   | 2016.07-2016.08 |   16    |        11        |

## Data

- adj_filename: path of the adjacency matrix file
- graph_signal_matrix_filename: path of graph signal matrix file
- num_of_vertices: number of vertices
- points_per_hour: points per hour, in our dataset is 12
- num_for_predict: points to predict, in our model is 12

# Data preprocessing

Weather is the preprocessing of weather data, mainly including the processing of raw data files for weather data, and adjusting the time slice of weather data to the same dimension as traffic data

# One-hot

**skyc1、skyc2**                                                                                                                                       **wx-codes**

| Feature code | One-hot vector        | &emsp;&emsp;&emsp;&emsp; | Feature code | One-hot vector          |
| ------------ | --------------------- | ------------------------ | ------------ | ----------------------- |
| CLR          | [  1 , 0, 0, 0, 0, 0] | &emsp;&emsp;&emsp;&emsp; | HZ           | [  1, 0, 0, 0, 0, 0, 0] |
| FEW          | [  0, 1, 0, 0, 0, 0]  | &emsp;&emsp;&emsp;&emsp; | RA           | [  0, 1, 0, 0, 0, 0, 0] |
| VV           | [  0, 0, 1, 0, 0, 0]  | &emsp;&emsp;&emsp;&emsp; | BR           | [  0, 0, 1, 0, 0, 0, 0] |
| SCT          | [  0, 0, 0, 1, 0, 0]  | &emsp;&emsp;&emsp;&emsp; | RA BR        | [  0, 0, 0, 1, 0, 0, 0] |
| BKN          | [  0, 0, 0, 0, 1, 0]  | &emsp;&emsp;&emsp;&emsp; | BR           | [  0, 0, 0, 0, 1, 0, 0] |
| OVC          | [  0, 0, 0, 0, 0, 1]  | &emsp;&emsp;&emsp;&emsp; | BCFG         | [  0, 0, 0, 0, 0, 1, 0] |
|              |                       | &emsp;&emsp;&emsp;&emsp; | BCFG         | [  0, 0, 0, 0, 0, 0, 1] |

## Training

- model_name: ASTGCN or MSTGCN
- ctx: set ctx = cpu, or set gpu-0, which means the first gpu device
- optimizer: sgd, RMSprop, adam, see [this page](https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#the-mxnet-optimizer-package) for more optimizer
- learning_rate: float, like 0.0001
- epochs: int, epochs to train
- batch_size: int
- num_of_weeks: int, how many weeks' data will be used
- num_of_days: int, how many days' data will be used
- num_of_hours: int, how many hours' data will be used
- K: int, K-order chebyshev polynomials will be used
- merge: int, 0 or 1, if merge equals 1, merge training set and validation set to train model
- prediction_filename: str, if you specify this parameter, it will save the prediction of current testing set into this file
- params_dir: the folder for saving parameters

# Traffic block



<img src="https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/%E4%BA%A4%E9%80%9A%E6%A8%A1%E5%9D%97.jpg" width="450" height="500" alt="交通模块" />

# Weather block

<img src="https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/%E5%A4%A9%E6%B0%94%E6%A8%A1%E5%9D%97.jpg" width="600" height="350" alt="天气模块" />

