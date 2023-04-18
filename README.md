# 1、Traffic Flow Prediction Architecture: AFFGCN
Spatial-temporal graph convolution network based on multimodal feature fusion, Combining weather data to predict traffic flow: A Spatiotemporal Graph Convolutional Network Method Based on Feature Fusion



<span style="font-size: 22px;">**<u>A</u>ttention <u>F</u>eature <u>F</u>usion base on spatial-temporal <u>G</u>raph <u>C</u>onvolutional <u>N</u>etwork（AFFGCN）**</span>![模型总体架构](https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/%E6%A8%A1%E5%9E%8B%E6%80%BB%E4%BD%93%E6%9E%B6%E6%9E%84.jpg)

# 2、Traffic dataset

Download PEMS04 and PEMS08 datasets provided by [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data)

![traffic dataset](https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/111.jpg)

| district |  Time interval  | detector |       feature       |
| :------: | :-------------: | :------: | :-----------------: |
| PEMS-08  | 2018.01-2018.02 |   307    | flow, occupy, speed |
| PEMS-04  | 2016.07-2016.08 |   170    | flow, occupy, speed |

# 3、Weather Dataset

The weather dataset for this article is from [the Iowa Atmospheric Observatory website](http://mesonet.agron.iastate.edu/request/download.phtml) at Iowa State University in the United States, which records historical weather data for most parts of the world. You can download the corresponding dataset by selecting seven steps on the website, including country and region, weather characteristics, and time and date. The downloaded weather data is a txt file. By manually converting it into a CSV file and setting the separator, you can better observe the initial state of the weather data.

| identifier | name           | significance                                                 |
| ---------- | -------------- | ------------------------------------------------------------ |
| 1          | valid          | Observation timestamp                                        |
| 2          | tmpf           | Fahrenheit temperature at a height of 2 meters               |
| 3          | dwpf           | Dew point temperature                                        |
| 4          | relh           | Relative humidity (%)                                        |
| 5          | drct           | Wind direction in degrees                                    |
| 6          | sknt           | Wind speed in knots                                          |
| ...        | ...            | ...                                                          |
| 24         | peak_wind_gust | Peak gust wind value                                         |
| 25         | peak_wind_drct | Peak gust direction                                          |
| 26         | peak_wind_time | Peak gust time                                               |
| 27         | feel           | Apparent temperature in Fahrenheit (wind cold or heat index) |

To match weather data that matches traffic data, download the corresponding weather data from the mesonet website (Iowa State University's Iowa Atmospheric Observatory network) based on the time and location of the traffic dataset. Firstly, the order of the weather time series corresponding to the time stamp is not processed. Features 15, 19, 24, 25, and 26 are not selected without data collection. Features 21, 22, and 23 are not selected due to the absence of ice accumulation in the time and location corresponding to the traffic dataset. Finally, the time stamp and 17 weather features remain.



![weather dataset](https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/21.jpg)

|        district        |  Time interval  | one-hot | Numeric Features |
| :--------------------: | :-------------: | :-----: | :--------------: |
| San Francisco Bay Area | 2018.01-2018.02 |   25    |        13        |
|  San Bernardino city   | 2016.07-2016.08 |   16    |        11        |

Weather  Dataset  One-hot  data:

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

## 4、Argparse

- adj_filename: path of the adjacency matrix file
- graph_signal_matrix_filename: path of graph signal matrix file
- num_of_vertices: number of vertices
- points_per_hour: points per hour, in our dataset is 12
- num_for_predict: points to predict, in our model is 12

# 5、Training

- model_name: ASTGCN
- ctx: set ctx = cpu, or set gpu-0, which means the first gpu device
- optimizer: sgd, RMSprop, adam, see [this page](https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#the-mxnet-optimizer-package) for more optimizer
- learning_rate: float, like 0.0001
- epochs: int, epochs to train
- batch_size: int
- K: int, K-order chebyshev polynomials will be used
- merge: int, 0 or 1, if merge equals 1, merge training set and validation set to train model
- prediction_filename: str, if you specify this parameter, it will save the prediction of current testing set into this file
- params_dir: the folder for saving parameters

# 6、Traffic block



<img src="https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/%E4%BA%A4%E9%80%9A%E6%A8%A1%E5%9D%97.jpg" width="450" height="500" alt="交通模块" />

# 7、Weather block

<img src="https://raw.githubusercontent.com/yanganYNU/AFFGCN/main/paper/images/%E5%A4%A9%E6%B0%94%E6%A8%A1%E5%9D%97.jpg" width="600" height="350" alt="天气模块" />

