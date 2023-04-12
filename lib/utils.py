"""
coding:utf-8
@Time    : 2022/7/12 5:21
@Author  : Alex-杨安
@FileName: utils.py
@Software: PyCharm
"""
import torch
import numpy as np
import csv
from scipy.sparse.linalg import eigs
from .metrics import mean_absolute_error, mean_squared_error, masked_mape_np


def normalization(train, val, test):
    # print('np.shape(train)=',np.shape(train))
    # print('np.shape(val)=', np.shape(val))
    # print('np.shape(test)=', np.shape(test),'\n')
    #
    # print('train.shape=',train.shape)
    # print('val.shape=',val.shape)
    # print('test.shape=',test.shape,'\n')
    #
    # print('train.shape[0]=', train.shape[0])
    # print('train.shape[1:]=', train.shape[1:])
    # print('train.shape[2]=', train.shape[2],'\n')
    #
    # print('val.shape[0]=', val.shape[0])
    # print('val.shape[1]=', val.shape[1])
    # print('val.shape[2]=', val.shape[2],'\n')
    #
    # print('test.shape[0]=', test.shape[0])
    # print('test.shape[1]=', test.shape[1])
    # print('test.shape[2]=', test.shape[2],'\n')
    # train、val、test分别都是四个维度的数据(8287, 170, 3, 24)、(2763, 170, 3, 24)、(2763, 170, 3, 24)
    # shape[1:]是1、2、3维也就是第2、3、4维的数据，assert判断train、val、test三个数据集的第2、3、4维的维度是否相同
    # assert的意思为判断条件为真则执行，不为真则抛出异常
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)  # wd: ??
    val_norm = normalize(val)
    test_norm = normalize(test)

    if len(train.shape) == 4:
        train_norm = (train_norm).transpose(0, 2, 1, 3)
        val_norm = (val_norm).transpose(0, 2, 1, 3)
        test_norm = (test_norm).transpose(0, 2, 1, 3)

    return {'mean': mean, 'std': std}, train_norm, val_norm, test_norm


def read_and_generate_dataset(tf_filename, wx_filename, num_of_index, merge=False):
    tf_data = np.load(tf_filename)['data']
    wx_data = np.load(wx_filename, allow_pickle=True)['arr_0']
    sequence_length = tf_data.shape[0]
    all_samples = []
    for index in range(sequence_length):
        start_index = index + num_of_index
        end_index = start_index + num_of_index
        if end_index > sequence_length:
            break
        samples_indices = [(index, start_index), (start_index, end_index)]

        tf_samples = np.concatenate([tf_data[i: j] for i, j in samples_indices], axis=0)
        # print(tf_samples.shape)
        wx_samples = np.concatenate([wx_data[i: j] for i, j in samples_indices], axis=0)
        target_tf = tf_data[index: index + num_of_index]
        target_wx = wx_data[index: index + num_of_index]
        # print(tf_samples.shape,wx_samples.shape,target_tf.shape,target_wx.shape)
        all_samples.append((
            np.expand_dims(tf_samples, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(wx_samples, axis=0).transpose((0, 2, 1)),
            np.expand_dims(target_wx, axis=0).transpose((0, 2, 1))[:, :, :],
            np.expand_dims(target_tf, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
        ))
        # print(tf_samples.shape)
        # print(len(all_samples))
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)
    if not merge:
        # 训练集，split_line1为60%的分割线，也就是将数据集的0-60%部分为训练集
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        # merge=True ，将20%的验证集也合并一起作为训练集
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_tf, train_wx, train_target_wx, train_target = training_set
    val_tf, val_wx, val_target_wx, val_target = validation_set
    test_tf, test_wx, test_target_wx, test_target = testing_set
    # 分别打印出（训练集、验证集、测试集）的（周、日、邻近）的数据维度
    print('training data: traffic: {}, weather: {}, target: {}'.format(
        train_tf.shape, train_wx.shape, train_target_wx.shape, train_target.shape))
    print('validation data: traffic: {}, weather: {}, target: {}'.format(
        val_tf.shape, val_wx.shape, val_target_wx.shape, val_target.shape))
    print('testing data: traffic: {}, weather: {}, target: {}'.format(
        test_tf.shape, test_wx.shape, test_target_wx.shape, test_target.shape))

    train_onehot = train_wx[:, 13:38]
    val_onehot = val_wx[:, 13:38]
    test_onehot = test_wx[:, 13:38]

    train_wx = train_wx[:, 0:13]
    val_wx = val_wx[:, 0:13]
    test_wx = test_wx[:, 0:13]

    (tf_stats, train_tf_norm, val_tf_norm, test_tf_norm) = normalization(train_tf, val_tf, test_tf)
    (wx_stats, train_wx_norm, val_wx_norm, test_wx_norm) = normalization(train_wx, val_wx, test_wx)

    train_wx_norm = np.append(train_wx_norm, train_onehot, axis=1)
    val_wx_norm = np.append(val_wx_norm, val_onehot, axis=1)
    test_wx_norm = np.append(test_wx_norm, test_onehot, axis=1)

    all_data = {
        'train': {
            'traffic': train_tf_norm,
            'weather': train_wx_norm,
            'target_wx': train_target_wx,
            'target': train_target,
        },
        'val': {
            'traffic': val_tf_norm,
            'weather': val_wx_norm,
            'target_wx': val_target_wx,
            'target': val_target,
        },
        'test': {
            'traffic': test_tf_norm,
            'weather': test_wx_norm,
            'target_wx': test_target_wx,
            'target': test_target,
        },
        'stats': {
            'traffic': tf_stats,
            'weather': wx_stats
        }
    }


    return all_data


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1
    return A


def scaled_Laplacian(W):
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def compute_val_loss(net, val_loader, loss_function, supports, device, epoch, alpha):
    '''
    compute mean loss on validation set

    Parameters
    ----------
    net: model

    val_loader: gluon.data.DataLoader

    loss_function: func

    epoch: int, current epoch

    '''
    net.eval()
    with torch.no_grad():
        tmp = []
        for index, (val_tf, val_wx, val_true_wx, val_true_tf) in enumerate(val_loader):
            val_tf = val_tf.to(device)
            val_wx = val_wx.to(device)
            val_true_wx = val_true_wx.to(device)
            val_true_tf = val_true_tf.to(device)
            out, _, _, out_wx = net(val_tf, val_wx, supports)
            l_tf = loss_function(out, val_true_tf)
            l_wx = loss_function(out_wx, val_true_wx)
            l = alpha * l_tf + (1 - alpha) * l_wx
            tmp.append(l.item())

        validation_loss = sum(tmp) / len(tmp)

        print('epoch: %s, validation loss: %.2f' % (epoch, validation_loss))
        return validation_loss


def predict(net, test_loader, supports, device):
    '''
    predict

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    Returns
    ----------
    prediction: np.ndarray,
                shape is (num_of_samples, num_of_vertices, num_for_predict)

    '''
    net.eval()
    with torch.no_grad():
        prediction = []
        for index, (test_tf, test_wx, test_t_wx, test_t) in enumerate(test_loader):
            test_tf = test_tf.to(device)
            test_wx = test_wx.to(device)

            output, _, _, _ = net(test_tf, test_wx, supports)
            prediction.append(output.cpu().detach().numpy())

        for index, (test_tf, test_wx, test_t_wx, test_t) in enumerate(test_loader):
            test_tf = test_tf.to(device)
            test_wx = test_wx.to(device)

            _, spatial_at, temporal_at, _ = net(test_tf, test_wx, supports)
            spatial_at = spatial_at.cpu().detach().numpy()
            temporal_at = temporal_at.cpu().detach().numpy()
            break

        prediction = np.concatenate(prediction, 0)
        return prediction, spatial_at, temporal_at


def evaluate(net, test_loader, true_value, supports, device, epoch):
    '''
    compute MAE, RMSE, MAPE scores of the prediction
    for 3, 6, 12 points on testing set

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    true_value: np.ndarray, all ground truth of testing set
                shape is (num_of_samples, num_for_predict, num_of_vertices)

    num_of_vertices: int, number of vertices

    epoch: int, current epoch

    '''
    net.eval()
    with torch.no_grad():
        prediction, _, _ = predict(net, test_loader, supports, device)

        # print(prediction.shape)
        # prediction = (prediction.transpose((0, 2, 1))
        #        .reshape(prediction.shape[0], -1))
        for i in [3, 6, 12]:
            print('current epoch: %s, predict %s points' % (epoch, i))

            mae = mean_absolute_error(true_value[:, :, 0:i],
                                      prediction[:, :, 0:i])
            rmse = mean_squared_error(true_value[:, :, 0:i],
                                      prediction[:, :, 0:i]) ** 0.5
            mape = masked_mape_np(true_value[:, :, 0:i],
                                  prediction[:, :, 0:i], 0)

            print('MAE: %.2f' % mae)
            print('RMSE: %.2f' % rmse)
            print('MAPE: %.2f' % mape)
