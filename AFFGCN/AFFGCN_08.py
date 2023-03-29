"""
coding:utf-8
@Time    : 2022/7/9 17:20
@Author  : Alex-杨安
@FileName: train.py
@Software: PyCharm
"""
import os
import shutil
from time import time
from datetime import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from lib.utils import scaled_Laplacian, get_adjacency_matrix
from lib.utils import compute_val_loss, evaluate, predict
from lib.utils import read_and_generate_dataset
from lib.model import AFFGCN as model

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.99, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--length', type=int, default=60, help='Size of temporal : 6')
parser.add_argument("--force", type=str, default=True,
                    help="remove params dir", required=False)
parser.add_argument("--data_name", type=str, default=8,
                    help="the number of data documents [8/4]", required=False)
parser.add_argument('--num_point', type=int, default=170, help='road Point Number [170/307] ', required=False)
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate [0.97/0.92]')

FLAGS = parser.parse_args()
f = FLAGS.data_name
decay = FLAGS.decay

adj_filename = 'data/PEMS0%s/distance.csv' % f
tf_filename = 'data/PEMS0%s/pems0%s.npz' % (f, f)
wx_filename = 'data/PEMS0%s/weather.npz' % f

Length = FLAGS.length
batch_size = FLAGS.batch_size

num_nodes = FLAGS.num_point
epochs = FLAGS.max_epoch

learning_rate = FLAGS.learning_rate
optimizer = FLAGS.optimizer
num_of_index = 12
num_of_hours = 2
num_of_vertices = FLAGS.num_point
num_tf_features = 3
alpha = 0.9
# time_size=24
merge = False
model_name = 'AFFGCN_0%s' % f
params_dir = 'result/exp/AFFGCN'
prediction_path = 'result/prediction/AFFGCN_0%s' % f
wdecay = 0.00
device = torch.device(FLAGS.device)

adj = get_adjacency_matrix(adj_filename, num_nodes)
adjs = scaled_Laplacian(adj)
supports = (torch.tensor(adjs)).type(torch.float32).to(device)

print('Model is %s' % (model_name))
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)

if os.path.exists(params_path) and not FLAGS.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s' % (params_path))

if __name__ == "__main__":
    print("Reading data...")
    all_data = read_and_generate_dataset(tf_filename, wx_filename, num_of_index, merge)

    # print(all_data['train']['traffic'][0][0][:][0])
    true_value = all_data['test']['target']
    print('true_value.shape=', true_value.shape)

    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['traffic']),
            torch.Tensor(all_data['train']['weather']),
            torch.Tensor(all_data['train']['target_wx']),
            torch.Tensor(all_data['train']['target']),
        ),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['traffic']),
            torch.Tensor(all_data['val']['weather']),
            torch.Tensor(all_data['val']['target_wx']),
            torch.Tensor(all_data['val']['target']),
        ),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['traffic']),
            torch.Tensor(all_data['test']['weather']),
            torch.Tensor(all_data['test']['target_wx']),
            torch.Tensor(all_data['test']['target']),
        ),
        batch_size=batch_size,
        shuffle=False
    )
    # save Z-score mean and std
    stats_data = {}
    for type_ in ['traffic', 'weather']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean']
        stats_data[type_ + '_std'] = stats['std']

    np.savez_compressed(
        os.path.join(params_path, 'stats_data'),
        **stats_data
    )
    loss_function = nn.MSELoss()

    net = model(c_in=num_tf_features, c_out=64, num_wx_features=27, num_for_predict=12,
                num_nodes=num_nodes, time_size=24, K=3, Kt=3)
    net.to(device)  # to cuda
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
    compute_val_loss(net, val_loader, loss_function, supports, device, epoch=0,alpha = alpha)
    evaluate(net, test_loader, true_value, supports, device, epoch=0)

    his_loss = []
    train_time = []
    for epoch in range(1, epochs + 1):
        train_l = []
        start_time_train = time()
        for train_tf, train_wx, train_true_wx, train_true_tf, in train_loader:
            train_tf = train_tf.to(device)
            train_wx = train_wx.to(device)
            train_true_tf = train_true_tf.to(device)
            train_true_wx = train_true_wx.to(device)
            net.train()  # train pattern
            optimizer.zero_grad()  # grad to 0

            out, _, _, out_wx = net(train_tf, train_wx, supports)
            l_tf = loss_function(out, train_true_tf)
            l_wx = loss_function(out_wx, train_true_wx)
            loss = alpha * l_tf + (1 - alpha) * l_wx
            # backward p
            loss.backward()

            # update parameter
            optimizer.step()

            training_loss = loss.item()
            train_l.append(training_loss)
        scheduler.step()
        end_time_train = time()
        train_l = np.mean(train_l)
        print('epoch step: %s, training loss: %.2f, time: %.2fs'
              % (epoch, train_l, end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)

        # compute validation loss
        valid_loss = compute_val_loss(net, val_loader, loss_function, supports, device, epoch,alpha)
        his_loss.append(valid_loss)

        # evaluate the model on testing set
        evaluate(net, test_loader, true_value, supports, device, epoch)

        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name,
                                                                  epoch, str(round(valid_loss, 2))))

        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % (params_filename))

    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))

    bestid = np.argmin(his_loss)

    print("The valid loss on best model is epoch%s_%s" % (str(bestid + 1), str(round(his_loss[bestid], 4))))
    best_params_filename = os.path.join(params_path,
                                        '%s_epoch_%s_%s.params' % (model_name,
                                                                   str(bestid + 1), str(round(his_loss[bestid], 2))))

    net.load_state_dict(torch.load(best_params_filename))
    start_time_test = time()
    prediction, spatial_at, temporal_at = predict(net, test_loader, supports, device)
    end_time_test = time()
    evaluate(net, test_loader, true_value, supports, device, epoch)
    test_time = np.mean(end_time_test - start_time_test)
    print("Test time: %.2f" % test_time)

    np.savez_compressed(
        os.path.normpath(prediction_path),
        prediction=prediction,
        spatial_at=spatial_at,
        temporal_at=temporal_at,
        ground_truth=all_data['test']['target']
    )
print(timestamp)
