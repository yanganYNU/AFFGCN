"""
coding:utf-8
@Time    : 2022/7/12 5:21
@Author  : Alex-杨安
@FileName: model.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv2d, Parameter, LayerNorm, Conv1d, BatchNorm1d
from torch.autograd import Variable


# 1、空间注意力层 当执行GCN时，我们将邻接矩阵A和空间注意力矩阵S结合起来动态调整节点之间的权重。
class SATT(nn.Module):
    # def __init__(self, c_in='3', num_nodes='170/307', tem_size='24/12/24'):
    def __init__(self, c_in, num_nodes, tem_size):
        # print('c_in=',c_in)
        super(SATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)

        # nn.Parameter 一组可训练参数
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        # 对self.w的参数进行初始化，且服从Xavier均匀分布

        self.b = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        # 全为0，因此不存在初始化

        self.v = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        # 对self.v的参数进行初始化，且服从Xavier均匀分布

    # seq：序列
    def forward(self, seq):
        # print('seq的维度：',seq.shape)，seq哪里来的？

        c1 = seq
        f1 = self.conv1(c1).squeeze(1)  # batch_size,num_nodes,length_time
        # print('f1的维度：',f1.shape)

        c2 = seq.permute(0, 3, 1, 2)  # b,c,n,l->b,l,n,c
        # print('c2的维度：', c2.shape)

        f2 = self.conv2(c2).squeeze(1)  # b,c,n
        # print('f2的维度：', f2.shape)

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        # print('logits的维度：', logits.shape)
        logits = torch.matmul(self.v, logits)
        # print('logits的维度：', logits.shape, '\n')
        ##normalization
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        # print('coefs=',coefs.shape)
        return coefs


# 2、图卷积层 具有空间注意分数的K阶chebyshev图卷积
class cheby_conv_ds(nn.Module):
    def __init__(self, c_in, c_out, K):
        super(cheby_conv_ds, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj, ds):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L0 = torch.eye(nNode).cuda()
        L1 = adj

        L = ds * adj
        I = ds * torch.eye(nNode).cuda()
        Ls.append(I)
        Ls.append(L)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            L3 = ds * L2
            Ls.append(L3)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        # print('out=', out.shape)
        return out

    ###ASTGCN_block


# 3、时间注意力层
class TATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, seq):
        # print('seq的维度：', seq.shape)
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        # print('c1的维度：', c1.shape)
        f1 = self.conv1(c1).squeeze(1)  # b,l,n
        # print('f1的维度：', f1.shape)

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        # print('c2的维度：', c2.shape)
        f2 = self.conv2(c2).squeeze(1)  # b,c,l
        # print('f2的维度：', f2.shape)

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        # print('logits的维度：', logits.shape)
        logits = torch.matmul(self.v, logits)
        # print('logits的维度：', logits.shape, '\n')
        ##normalization
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        # print('coefs=', coefs.shape)
        return coefs


# 4、时空块 整体时空块的搭建，用到了前面的1、2、3
class ST_BLOCK_0(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_0, self).__init__()

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT = TATT(c_in, num_nodes, tem_size)
        self.SATT = SATT(c_in, num_nodes, tem_size)
        self.dynamic_gcn = cheby_conv_ds(c_in, c_out, K)
        self.K = K
        self.time_conv = Conv2d(c_out, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        T_coef = self.TATT(x)
        T_coef = T_coef.transpose(-1, -2)
        x_TAt = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        S_coef = self.SATT(x)  # B x N x N

        spatial_gcn = self.dynamic_gcn(x_TAt, supports, S_coef)
        spatial_gcn = torch.relu(spatial_gcn)
        time_conv_output = self.time_conv(spatial_gcn)
        out = self.bn(torch.relu(time_conv_output + x_input))
        # print('out=', out.shape)
        # print('ST_BLOCK_0:supports=', supports[1][1:3])
        # print('ST_BLOCK_0:supports=', supports.shape,'\n')
        return out, S_coef, T_coef

    ###1


# 5、交通时空块堆叠
class tf_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(tf_block, self).__init__()
        self.block1 = ST_BLOCK_0(c_in, c_out, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK_0(c_out, c_out, num_nodes, tem_size, K, Kt)
        self.final_conv = Conv2d(tem_size, 12, kernel_size=(1, c_out), padding=(0, 0),
                                 stride=(1, 1), bias=True)
        self.w = Parameter(torch.zeros(num_nodes, 12), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

    # 前向传播
    def forward(self, x, supports):
        x, _, _ = self.block1(x, supports)
        x, d_adj, t_adj = self.block2(x, supports)
        x = x.permute(0, 3, 2, 1)
        # print('x.shape =',x.shape)
        x = self.final_conv(x).squeeze().permute(0, 2, 1)  # b,n,12
        # print('x.shape =', x.shape)
        x = x * self.w
        # print('x.shape =', x.shape)
        # print('ASTGCN_block:supports=', supports[1][1:3])
        # print('ASTGCN_block:supports=', supports.shape,'\n')
        return x, d_adj, t_adj


# 6、 天气模型网络
class wx_block(nn.Module):
    def __init__(self, num_wx_features, time_size, num_nodes, num_for_predict):
        super(wx_block, self).__init__()

        input_size = num_wx_features
        hidden_size = num_wx_features
        num_layers = 2
        self.GRU = nn.GRU(input_size, hidden_size, num_layers)

        in_channels = time_size
        out_channels = num_for_predict
        self.conv1 = Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_wx):
        x = x_wx.permute(2, 0, 1).contiguous()

        c_1, h_1 = self.GRU(x)
        c_1 = c_1.permute(1, 0, 2).contiguous()
        out = self.conv1(c_1).contiguous()

        out = out.permute(0, 2, 1).contiguous()
        return out


# Attentional Feature Fusion Spatial-Temporal Attention base Multi-Task Net
# 7、attention（天气，交通）
class ATTFF(nn.Module):
    def __init__(self, num_for_predict, num_nodes, num_wx_features):
        # 输入维度（batch_size, embed_dim, term_size）
        # 输出维度（batch_size, num_node, term_size）
        super(ATTFF, self).__init__()
        # in_features = num_nodes
        # out_features = num_nodes
        # self.linear1 = nn.Linear(in_features, out_features, bias=True)

        in_features = num_wx_features
        out_features = num_nodes
        self.linear2 = nn.Linear(in_features, out_features, bias=True)
        self.linear3 = nn.Linear(in_features, out_features, bias=True)

        embed_dim = num_for_predict
        num_heads = 4
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x, out_wx):  # (16,170,12)
        query = x
        key = self.linear2(out_wx.transpose(1, 2))
        value = self.linear3(out_wx.transpose(1, 2))

        query = query.transpose(0, 1)
        key = key.permute(2, 0, 1).contiguous()
        value = value.permute(2, 0, 1).contiguous()
        out, _ = self.attention(query, key, value)

        out = out.transpose(0, 1)
        return out


class AFFGCN(nn.Module):
    def __init__(self, c_in, c_out, num_wx_features, num_for_predict, num_nodes, time_size, K, Kt):
        super(AFFGCN, self).__init__()
        self.block_tf = tf_block(c_in, c_out, num_nodes, time_size, K, Kt)
        self.bn_tf = BatchNorm2d(c_in, affine=False)

        self.block_wx = wx_block(num_wx_features, time_size, num_nodes, num_for_predict)
        # self.bn_wx = BatchNorm1d(num_wx_features, affine=False)

        in_features = num_wx_features
        out_features = num_nodes
        self.linear_wx = nn.Linear(in_features, out_features, bias=True)

        self.out_attention = ATTFF(num_for_predict, num_nodes, num_wx_features)

    def forward(self, x_tf, x_wx, supports):
        # (batch_size, in_channels, H, W)
        # 交通输入：(batch_size, 3, 170, 24)
        # 交通输出：(batch_size, 170, 12)
        # 天气输入：(batch_size, 11, 24)
        # 天气输出：(batch_size, 170, 12)
        x_tf = self.bn_tf(x_tf)
        x_tf, d_adj_r, t_adj_r = self.block_tf(x_tf, supports)

        out_wx = self.block_wx(x_wx)
        att_wx = self.out_attention(x_tf, out_wx)
        out = att_wx + x_tf
        return out, d_adj_r, t_adj_r, out_wx
