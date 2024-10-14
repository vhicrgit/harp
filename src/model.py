from config import FLAGS
from saver import saver
from utils import MLP, _get_y_with_target, MLP_multi_objective, MultiAttension_MLP_multi_objective

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GlobalAttention, JumpingKnowledge, TransformerConv, GCNConv
from torch_geometric.nn import global_add_pool, BatchNorm
import torch.nn as nn

from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU, Embedding

from collections import OrderedDict, defaultdict
from torch_geometric.nn import TransformerConv
from typing import Optional

class MultiheadTransformer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 hidden_channels=64, 
                 num_heads=4,        
                 dropout: float = 0.,
                 edge_dim: Optional[int] = None,
                 resnet_enable = True):
        super().__init__()
        self.resnet_enable = resnet_enable
        self.num_heads = num_heads
        self.bn = BatchNorm(in_channels)
        self.residual = torch.nn.Identity()
        self.conv = TransformerConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, edge_dim=edge_dim)
        self.fc = Linear(hidden_channels * num_heads, out_channels)
        
        nn.init.xavier_normal_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x, edge_index, edge_attr=None):
        if self.resnet_enable:
            res = self.residual(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.fc(x)
        if self.resnet_enable:
            x += res
        return x


class NodeEmbedding(nn.Module):
    def __init__(self):
        super(NodeEmbedding, self).__init__()
        self.residual = torch.nn.Identity()
        # self.n_embed  = Embedding(num_embeddings=FLAGS.n_size, embedding_dim=FLAGS.n_embed_dim)
        # self.p_embed = Embedding(num_embeddings=FLAGS.p_size, embedding_dim=FLAGS.p_embed_dim)
        if FLAGS.numeric_embed_dim != 1:
            self.numeric_embed = Embedding(num_embeddings=FLAGS.numeric_size, embedding_dim=FLAGS.numeric_embed_dim)
        else:
            self.numeric_embed = torch.nn.Identity()
        self.i_embed = Embedding(num_embeddings=FLAGS.i_size, embedding_dim=FLAGS.i_embed_dim)
        # self.f_embed = Embedding(num_embeddings=FLAGS.f_size, embedding_dim=FLAGS.f_embed_dim)
        # self.b_embed = Embedding(num_embeddings=FLAGS.b_size, embedding_dim=FLAGS.b_embed_dim)
        total_dim = FLAGS.n_embed_dim + FLAGS.p_embed_dim + FLAGS.numeric_embed_dim + FLAGS.i_embed_dim + FLAGS.f_embed_dim + FLAGS.b_embed_dim
        self.fc = Linear(total_dim, 64)
    
    def forward(self, X):
        n, p, numeric, i, f, b = X[:,:5], X[:,5:9], X[:,9], X[:,10:93], X[:,93:101], X[:,101:]
        # n = self.n_embed(n.argmax(dim=1))
        # p = self.p_embed(p.argmax(dim=1))
        numeric = self.numeric_embed(numeric.to(int))
        # numeric = numeric.view(-1, FLAGS.numeric_embed_dim)
        i = self.i_embed(i.argmax(dim=1))
        # f = self.f_embed(f.argmax(dim=1))
        # b = self.b_embed(b.argmax(dim=1))
        # out_embed = self.fc(torch.cat((n, p, numeric, i, f, b), dim=1).to(torch.float))
        x = torch.cat((numeric, i), dim=1).to(torch.float)
        res = self.residual(x)
        x = self.fc(x)
        out_embed = x + res
        return out_embed
    

class EdgeEmbedding(nn.Module):
    def __init__(self):
        super(EdgeEmbedding, self).__init__()
        self.p_embed = Embedding(num_embeddings=FLAGS.edge_p_size, embedding_dim=FLAGS.edge_p_embed_dim)
        self.f_embed = Embedding(num_embeddings=FLAGS.edge_f_size, embedding_dim=FLAGS.edge_f_embed_dim)
        total_dim = FLAGS.edge_p_embed_dim + FLAGS.edge_f_embed_dim
        self.fc = Linear(total_dim, FLAGS.D)
    
    def forward(self, edge):
        f, p = edge[:,:7], edge[:,7:]
        p = self.p_embed(p.argmax(dim=1))
        f = self.f_embed(f.argmax(dim=1))
        out_embed = self.fc(torch.cat((f, p), dim=1).to(torch.float))
        return out_embed


class Net(nn.Module):
    def __init__(self, in_channels, edge_dim = 0, init_pragma_dict = None, task = FLAGS.task, num_layers = FLAGS.num_layers, D = FLAGS.D, target = FLAGS.target):
        super(Net, self).__init__()
        
        
        if FLAGS.enable_embedding:
            if FLAGS.bad_embedding:
                self.node_embedding = Embedding(num_embeddings=in_channels, embedding_dim=64)
                self.edge_embedding = Embedding(num_embeddings=edge_dim, embedding_dim=64)            
            else:
                self.node_embedding = NodeEmbedding()
                self.edge_embedding = EdgeEmbedding()
            resnet_enable = True
            in_channels = FLAGS.node_embed_dim
            edge_dim = FLAGS.edge_embed_dim
        else:
            resnet_enable = False
            
        
        self.MLP_version = 'multi_obj'  if len(FLAGS.target) > 1 else  'single_obj'
        if FLAGS.gnn_type == 'gat':
            conv_clconv_class = TransformerConvass = GATConv
        elif FLAGS.gnn_type == 'gcn':
            conv_class = GCNConv
        elif FLAGS.gnn_type == 'transformer':
            conv_class = TransformerConv
        elif FLAGS.gnn_type == 'multihead_transformer':
            conv_class = MultiheadTransformer
        else:
            raise NotImplementedError()
        
        if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
            self.conv_first = conv_class(in_channels, D, edge_dim=edge_dim, dropout=FLAGS.dropout)
        elif FLAGS.encode_edge and FLAGS.gnn_type == 'multihead_transformer':
            self.conv_first = conv_class(in_channels, D, edge_dim=edge_dim, dropout=FLAGS.dropout, num_heads=4, resnet_enable=resnet_enable)
        else:
            self.conv_first = conv_class(in_channels, D)

        self.conv_layers = nn.ModuleList()

        self.num_conv_layers = num_layers - 1
        num_layers += FLAGS.gnn_layer_after_MLP
        for i in range(num_layers - 1):
            if i == num_layers - 2:
                heads = 4
            # elif i == num_layers - 3 or i == num_layers - 4:
            #     heads = 2
            else:
                heads = 4
            if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                conv = conv_class(D, D, edge_dim=edge_dim, dropout=FLAGS.dropout)
            elif FLAGS.encode_edge and FLAGS.gnn_type == 'multihead_transformer':
                conv = conv_class(D, D, edge_dim=edge_dim, dropout=FLAGS.dropout, num_heads=heads)
            else:
                conv = conv_class(D, D)
            self.conv_layers.append(conv)

        if FLAGS.gae_T: # graph auto encoder
            if FLAGS.separate_T:
                self.gae_transform_T = nn.ModuleDict()
                for gname, feat_dim in init_pragma_dict.items():
                    self.gae_transform_T['all'] = Linear(feat_dim[1], D // 8) 
                channels = [D // 2, D // 4]
                self.decoder_T = MLP(D, D // 8,
                            activation_type=FLAGS.activation,
                            hidden_channels=channels,
                            num_hidden_lyr=len(channels))
        if FLAGS.gae_P:
            out_channels = in_channels
            if FLAGS.input_encode:
                self.gate_input = Linear(in_channels, 2 * D) ## encode input one-hot representation
                out_channels = 2 * D
            
            if FLAGS.decoder_type == 'type1':
                decoder_arch = []
            elif FLAGS.decoder_type == 'type2':
                decoder_arch = [D, 2 * D, out_channels]
            self.decoder_P = MLP(D, out_channels, activation_type = FLAGS.activation,
                            hidden_channels = decoder_arch,
                            num_hidden_lyr = len(decoder_arch))
            if FLAGS.decoder_type == 'None':
                for name, param in self.decoder_P.named_parameters():
                    print(name)
                    param.requires_grad = False
        if FLAGS.gae_T or FLAGS.gae_P:
            self.gae_sim_function = nn.CosineSimilarity()
            self.gae_loss_function = nn.CosineEmbeddingLoss()

        self.jkn = JumpingKnowledge(FLAGS.jkn_mode, channels=D, num_layers=2)

        self.task = task

        if task == 'regression':
            self.out_dim = 1
            self.MLP_out_dim = 1
            self.loss_function = nn.MSELoss()
        else:
            self.out_dim = 2
            self.MLP_out_dim = 2
            self.loss_function = nn.CrossEntropyLoss()
        
        if FLAGS.node_attention:
            if FLAGS.multi_attention:
                # program and pseudo attention
                self.gate_nn_P = self.node_att_gate_nn(D)
                # self.glob_P = MyGlobalAttention(self.gate_nn_P, None)
                self.gate_nn_pseudo_B = self.node_att_gate_nn(D)
                # self.glob_pseudo_B = MyGlobalAttention(self.gate_nn_pseudo_B, None)
            else:
                if FLAGS.separate_T:
                    self.gate_nn_T = self.node_att_gate_nn(D)
                    self.glob_T = MyGlobalAttention(self.gate_nn_T, None)
                if FLAGS.separate_P:
                    self.gate_nn_P = self.node_att_gate_nn(D)
                    self.glob_P = MyGlobalAttention(self.gate_nn_P, None)
                if FLAGS.separate_pseudo: ## for now, only pseudo node for block
                    self.gate_nn_pseudo_B = self.node_att_gate_nn(D)
                    self.glob_pseudo_B = MyGlobalAttention(self.gate_nn_pseudo_B, None)
                if FLAGS.separate_icmp:
                    self.gate_nn_icmp = self.node_att_gate_nn(D)
                    self.glob_icmp = MyGlobalAttention(self.gate_nn_icmp, None)
        
        
        if 'regression' in self.task:
            _target_list = target
            if not isinstance(FLAGS.target, list):
                _target_list = [target]
            self.target_list = [t for t in _target_list]
        else:
            self.target_list = ['perf']
        
        if FLAGS.node_attention:
            if FLAGS.multi_attention:
                dim = FLAGS.separate_P + FLAGS.separate_pseudo
                in_D = dim * D
            else:
                dim = FLAGS.separate_T + FLAGS.separate_P + FLAGS.separate_pseudo + FLAGS.separate_icmp
                in_D = dim * D
        else:
            in_D = D
        if D > 64:
            hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
        else:
            hidden_channels = [D // 2, D // 4, D // 8]
        if self.MLP_version == 'single_obj':
            self.MLPs = nn.ModuleDict()
            for target in self.target_list:
                self.MLPs[target] = MLP(in_D, self.MLP_out_dim, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels,
                                        num_hidden_lyr=len(hidden_channels))
        elif FLAGS.multi_attention:
            self.MLPs = MultiAttension_MLP_multi_objective(in_D, self.MLP_out_dim, self.gate_nn_P, self.gate_nn_pseudo_B, \
                                    activation_type=FLAGS.activation, \
                                    hidden_channels=hidden_channels, \
                                    objectives=self.target_list)
        else:
            self.MLPs = MLP_multi_objective(in_D, self.MLP_out_dim, activation_type=FLAGS.activation,
                                    hidden_channels=hidden_channels,
                                    objectives=self.target_list,
                                    num_common_lyr=FLAGS.MLP_common_lyr)
                
        ## pragma as MLP
        if FLAGS.pragma_as_MLP:
            self.pragma_as_MLP_list = FLAGS.pragma_as_MLP_list
            self.MLPs_per_pragma = nn.ModuleDict()
            for target in self.pragma_as_MLP_list:
                in_D = D + 1
                if target == 'parallel': in_D = D + 2 ## reduction/normal, factor
                hidden_channels, len_hidden_channels = None, 0
                if FLAGS.pragma_MLP_hidden_channels is not None:
                    hidden_channels = eval(FLAGS.pragma_MLP_hidden_channels)
                    len_hidden_channels = len(hidden_channels)
                self.MLPs_per_pragma[target] = MLP(in_D, D, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels, num_hidden_lyr=len_hidden_channels)
            if FLAGS.pragma_order == 'parallel_and_merge':
                in_D = D * len(self.pragma_as_MLP_list)
                hidden_channels = eval(FLAGS.merge_MLP_hidden_channels)
                
                self.MLPs_per_pragma['merge'] = MLP(in_D, D, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels, num_hidden_lyr=len(hidden_channels))

    def node_att_gate_nn(self, D):
        if FLAGS.node_attention_MLP:
            return MLP(D, 1,
                    activation_type=FLAGS.activation_type,
                    hidden_channels=[D // 2, D // 4, D // 8],
                    num_hidden_lyr=3)
        else:
            return Sequential(Linear(D, D), ReLU(), Linear(D, 1))

    def cal_gae_loss(self, encoded_g, decoded_out):
        target = torch.ones(len(encoded_g), device=FLAGS.device)  ## for similarity, use the negative form for dissimilarity
        target.requires_grad = False
        gae_loss = self.gae_loss_function(encoded_g, decoded_out, target)
        return gae_loss
        
    def mask_emb(self, out, non_zero_ids):
        out = out.permute((1, 0))
        out = out * non_zero_ids
        out = out.permute((1, 0))
        
        return out
    
    
    def apply_pragam_as_MLP(self, mlp_pragma, out, scope_nodes, X_pragma_per_node, ptype):
        if ptype == 'tile':
            pragma_option = X_pragma_per_node[:, 0].reshape(-1, 1)
        elif ptype == 'pipeline':
            pragma_option = X_pragma_per_node[:, 1].reshape(-1, 1)
        elif ptype == 'parallel':
            pragma_option = X_pragma_per_node[:, 2:4].reshape(-1, 2)
        elif ptype == 'merge':
            mlp_inp = X_pragma_per_node
        else:
            raise NotImplementedError()
            
        non_scope_nodes = torch.sub(1, scope_nodes)
        masked_emb = scope_nodes.ge(0.5)
        if ptype == 'merge':
            # mlp_out = mlp_pragma(mlp_inp[masked_emb])
            # out[masked_emb] = mlp_out
            mlp_out = mlp_pragma(self.mask_emb(mlp_inp, non_zero_ids=scope_nodes))
            out = self.mask_emb(out, non_zero_ids=non_scope_nodes) + self.mask_emb(mlp_out, non_zero_ids=scope_nodes)
        else:
            mlp_inp = torch.cat((out, pragma_option), dim=1)
            # mlp_out = mlp_pragma(mlp_inp[masked_emb])
            # out = torch.clone(out)
            # out[masked_emb] = mlp_out
            mlp_out = mlp_pragma(self.mask_emb(mlp_inp, non_zero_ids=scope_nodes))
            if FLAGS.pragma_order == 'sequential':    
                out = self.mask_emb(out, non_zero_ids=non_scope_nodes) + self.mask_emb(mlp_out, non_zero_ids=scope_nodes)
            elif FLAGS.pragma_order == 'parallel_and_merge':
                out = self.mask_emb(mlp_out, non_zero_ids=scope_nodes)
            else:
                raise NotImplementedError()
        
        return out
    
          
    def forward(self, data, inference_mode=False):
        x, edge_index, edge_attr, batch, pragmas = \
            data.x, data.edge_index, data.edge_attr, data.batch , data.pragmas
        if hasattr(data, 'kernel'):
            gname = data.kernel[0]
        if hasattr(data, 'X_pragma_per_node'):
            X_pragma_per_node = data.X_pragma_per_node
        outs = []
        out_dict = OrderedDict()
        if FLAGS.activation == 'relu':
            activation = F.relu
        elif FLAGS.activation == 'elu':
            activation = F.elu
        else:
            raise NotImplementedError()
        
        if FLAGS.enable_embedding:
            if FLAGS.bad_embedding:
                x = self.node_embedding(x.argmax(dim=1))
                edge_attr = self.edge_embedding(edge_attr.argmax(dim=1))
            else:
                x = self.node_embedding(x)
                edge_attr = self.edge_embedding(edge_attr)
        
        if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
            out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr))
        elif FLAGS.encode_edge and FLAGS.gnn_type == 'multihead_transformer':
            out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr))
        else:
            out = activation(self.conv_first(x, edge_index))

        outs.append(out)

        for i in range(self.num_conv_layers):
            conv = self.conv_layers[i]
            if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                out = conv(out, edge_index, edge_attr=edge_attr)
            elif FLAGS.encode_edge and FLAGS.gnn_type == 'multihead_transformer':
                out = conv(out, edge_index, edge_attr=edge_attr)
            else:
                out = conv(out, edge_index)
            if i != len(self.conv_layers) - 1:
                out = activation(out)
                
            outs.append(out)

        if FLAGS.jkn_enable:
            out = self.jkn(outs)
                
        ## pragma as MLP
        if FLAGS.pragma_as_MLP:
            in_merge = None
            for pragma in self.pragma_as_MLP_list:
                out_MLP = self.apply_pragam_as_MLP(self.MLPs_per_pragma[pragma], out, \
                                         data.X_pragmascopenids, X_pragma_per_node, pragma)
                if FLAGS.pragma_order == 'sequential':
                    out = out_MLP
                elif FLAGS.pragma_order == 'parallel_and_merge':
                    if in_merge is None: in_merge = out_MLP
                    else: in_merge = torch.cat((in_merge, out_MLP), dim=1)
                else:
                    raise NotImplementedError()
            ## the merge part
            if FLAGS.pragma_order == 'parallel_and_merge':
                out = self.apply_pragam_as_MLP(self.MLPs_per_pragma['merge'], out, \
                                         data.X_pragmascopenids, in_merge, 'merge')
                
            for i, conv in enumerate(self.conv_layers[self.num_conv_layers:]):
                if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
                    out = conv(out, edge_index, edge_attr=edge_attr)
                elif FLAGS.encode_edge and  FLAGS.gnn_type == 'multihead_transformer':
                    out = conv(out, edge_index, edge_attr=edge_attr)
                else:
                    out = conv(out, edge_index)
                if i != len(self.conv_layers) - 1:
                    out = activation(out)
        
        if FLAGS.node_attention and FLAGS.multi_attention:
            out_dict['emb_P'] = torch.tensor([])
            out_dict['emb_pseudo_b'] = torch.tensor([])
        elif FLAGS.node_attention:
            out_gnn = out
            out_g = None
            out_P, out_T = None, None
            if FLAGS.separate_P:
                if FLAGS.P_use_all_nodes:
                    out_P, node_att_scores_P = self.glob_P(out_gnn, batch)
                else:
                    out_P, node_att_scores_P = self.glob_P(out_gnn, batch, set_zeros_ids=data.X_contextnids)

                out_dict['emb_P'] = out_P
                out_g = out_P
                
            if FLAGS.separate_T:
                out_T, node_att_scores = self.glob_T(out_gnn, batch, set_zeros_ids=data.X_pragmanids)
                out_dict['emb_T'] = out_T
                if out_P is not None:
                    out_g = torch.cat((out_P, out_T), dim=1)
                else:
                    out_g = out_T
                    
            if FLAGS.separate_pseudo:
                out_pseudo_B, node_att_scores_pseudo = self.glob_pseudo_B(out_gnn, batch, set_zeros_ids=data.X_pseudonids)
                out_dict['emb_pseudo_b'] = out_pseudo_B
                if out_g is not None:
                    out_g = torch.cat((out_g, out_pseudo_B), dim=1)
                else:
                    out_g = out_pseudo_B   

            if FLAGS.separate_icmp:
                out_icmp, node_att_scores_icmp = self.glob_icmp(out_gnn, batch, set_zeros_ids=data.X_icmpnids)
                out_dict['emb_icmp'] = out_icmp
                if out_g is not None:
                    out_g = torch.cat((out_g, out_icmp), dim=1)
                else:
                    out_g = out_icmp             
            
            if not FLAGS.separate_P and not FLAGS.separate_T and not FLAGS.separate_pseudo:
                out_g, node_att_scores = self.glob_T(out_gnn, batch)
                out_dict['emb_T'] = out
                if FLAGS.subtask == 'visualize':
                    from saver import saver
                    saver.save_dict({'data': data, 'node_att_scores': node_att_scores},
                                    f'node_att.pickle')
            out = out_g
        else:
            out = global_add_pool(out, batch, data.X_pseudonids)
            out_dict['emb_T'] = out

        total_loss = 0
        gae_loss = 0
        if FLAGS.gae_T: # graph auto encoder
            assert FLAGS.separate_T
            gname = 'all'
            encoded_g = self.gae_transform_T[gname](pragmas)
            decoded_out = self.decoder_T(out_dict['emb_T'])
            gae_loss = self.cal_gae_loss(encoded_g, decoded_out)
        if FLAGS.gae_P:
            assert FLAGS.separate_P
            encoded_x = x
            if FLAGS.input_encode:
                encoded_x = self.gate_input(x)
            encoded_g = global_add_pool(encoded_x, batch) ## simple addition of node embeddings for gae
            
            if FLAGS.decoder_type == 'None': ## turn off autograd:
                decoded_out = self.decoder_P(out_dict['emb_P']).detach()
            else: 
                decoded_out = self.decoder_P(out_dict['emb_P']).to(FLAGS.device)
            # gae_loss = (self.gae_loss_function(encoded_g, decoded_out)).mean()
            gae_loss += self.cal_gae_loss(encoded_g, decoded_out)
        if FLAGS.gae_P or FLAGS.gae_T:
            total_loss += torch.abs(gae_loss)        
        
        out_embed = out        
        loss_dict = {}
        if self.MLP_version == 'multi_obj':
            if FLAGS.node_attention and FLAGS.multi_attention:
                # 此时输入应该是图，MLPs对每个任务执行一个global_attention，再分别传入MLP
                out_MLPs, att_dict = self.MLPs(out, batch)
            else:  
                out_MLPs, att_dict = self.MLPs(out_embed)
        for target_name in self.target_list:
            if self.MLP_version == 'multi_obj':
                out = out_MLPs[target_name]
            else:
                out = self.MLPs[target_name](out_embed)
            if not inference_mode:
                y = _get_y_with_target(data, target_name)
                if self.task == 'regression':
                    target = y.view((len(y), self.out_dim))
                    # print('target', target.shape)
                    if FLAGS.loss == 'RMSE':
                        loss = torch.sqrt(self.loss_function(out, target))
                        # loss = mean_squared_error(target, out, squared=False)
                    elif FLAGS.loss == 'MSE':
                        loss = self.loss_function(out, target) 
                    else:
                        raise NotImplementedError()
                    # print('loss', loss.shape)
                else:
                    target = y.view((len(y)))
                    loss = self.loss_function(out, target)
                loss_dict[target_name] = loss
                total_loss += loss
            out_dict[target_name] = out



        return out_dict, total_loss, loss_dict, gae_loss, att_dict
   
        
        
