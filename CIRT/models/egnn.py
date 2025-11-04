import torch
import torch.nn as nn
import torch.nn.functional as F


class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 34
        self.hidden_nf = hidden_nf

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2 + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 11 * 2, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.norm = nn.LayerNorm(11)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())


    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))

        # Calculate the number of samples in the batch
        batch_size = x.size(0) // (121 * 240)
        total_nodes = x.size(0)
        
        # Reshape for batch processing
        lat_agg = agg.reshape(batch_size, 121, 240, self.hidden_nf)
        lat_agg = torch.mean(lat_agg, dim=2, keepdim=True)  # Average over longitude
        lat_agg = lat_agg.repeat(1, 1, 240, 1).reshape(total_nodes, self.hidden_nf)

        # lon_agg = agg.reshape(batch_size, 121, 240, self.hidden_nf)
        # lon_agg = torch.mean(lon_agg, dim=1, keepdim=True)  # Average over latitude
        # lon_agg = lon_agg.repeat(1, 121, 1, 1).reshape(total_nodes, self.hidden_nf)

        global_agg = torch.mean(agg, dim=0, keepdim=True)
        global_agg = global_agg.repeat(total_nodes, 1)

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr, lat_agg], dim=1)
        else:
            agg = torch.cat([x, agg, lat_agg], dim=1)
        out = self.node_mlp(agg) + x
        return out, agg

    def coord_model(self, u, v, edge_index, radial, edge_feat):
        row, col = edge_index
        edge_feat = self.coord_mlp(edge_feat).reshape(edge_feat.size(0), 2, 11)
        radial = radial.unsqueeze(-1)
        wind = edge_feat * radial
        

        # edge_feat = self.coord_mlp(edge_feat).reshape(edge_feat.size(0), 2, 11)
        # radial = radial.unsqueeze(-1)
        # wind = edge_feat * radial
        # wind = torch.mean(edge_feat * radial, dim=-1)

        # trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg_u = unsorted_segment_mean(wind[:, 0, :], row, num_segments=u.size(0))
        agg_v = unsorted_segment_mean(wind[:, 1, :], row, num_segments=u.size(0))
        agg_u = torch.clamp(agg_u, min=-100, max=100)
        agg_v = torch.clamp(agg_v, min=-100, max=100)
        return agg_u, agg_v

    def coord2radial(self, edge_index, u, v):
        row, col = edge_index

        radial = torch.cat((u[col].unsqueeze(1), v[col].unsqueeze(1)), dim=1)
        col_speed = torch.norm(torch.stack((u[col], v[col])), dim=0)
        row_speed = torch.norm(torch.stack((u[row], v[row])), dim=0)
        rel_speed = torch.norm(torch.stack((u[row] - u[col], v[row] - v[col])), dim=0)
        col_dirt = torch.atan2(u[col], v[col])
        row_dirt = torch.atan2(u[row], v[row])
        # Add epsilon to prevent division by zero
        eps = 1e-8
        speed_product = col_speed * row_speed
        rel_dirt = (u[col] * u[row] + v[col] * v[row]) / (speed_product.clamp(min=eps))
        # col_speed = self.norm(col_speed)
        # row_speed = self.norm(row_speed)
        # rel_speed = self.norm(rel_speed)
        w_diff = torch.cat((rel_dirt, col_speed, row_speed), dim=1)
        # w_diff = rel_dirt

        return w_diff, radial

class E_GCL_vel(E_GCL):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn,
                       recurrent=recurrent, coords_weight=coords_weight, attention=attention, norm_diff=norm_diff, tanh=tanh)
        self.norm_diff = norm_diff
        self.edge_norm = nn.LayerNorm(hidden_nf)
        self.coord_mlp_u = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 22))
        
        self.coord_mlp_v = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 22))

        # self.coord_w = nn.Sequential(
        #     nn.Linear(hidden_nf + 22, hidden_nf),
        #     act_fn,
        #     nn.Linear(hidden_nf, 1))

    def forward(self, h, u, v, radial, edge_index, edge_attr=None, node_attr=None):
        row, col = edge_index
        w_diff, _ = self.coord2radial(edge_index, u, v)

        edge_feat = self.edge_model(h[row], h[col], w_diff, edge_attr)
        edge_feat = self.edge_norm(edge_feat)
        agg_u, agg_v = self.coord_model(u, v, edge_index, radial, edge_feat)

        # speed = torch.sqrt(u**2 + v**2)
        # dirt = torch.atan2(u, v)
        # coord_w = self.coord_w(torch.cat([h, speed, dirt], dim=1))

        u = agg_u 
        v = agg_v 

        # u = agg_u #+ self.coord_mlp_u(torch.cat([h, speed, dirt], dim=1)) * u
        # v = agg_v #+ self.coord_mlp_v(torch.cat([h, speed, dirt], dim=1)) * v
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, u, v, edge_feat
    
class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, output_dim, act_fn=nn.GELU(), n_layers=5, recurrent=False, norm_diff=False, tanh=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.pred_len = 2
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        # self.edge_embedding = nn.Linear(69, self.hidden_nf)
        self.output_dim = output_dim
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                    act_fn=act_fn, recurrent=recurrent,
                                                    norm_diff=norm_diff, tanh=tanh))
            
        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf * 4),
                              act_fn,
                            #   nn.Linear(hidden_nf, hidden_nf),
                            #   act_fn,
                              nn.Linear(hidden_nf * 4, output_dim * self.pred_len))

        # self.coord_mlp_u = nn.Sequential(
        #     nn.Linear(hidden_nf, hidden_nf),
        #     act_fn,
        #     nn.Linear(hidden_nf, 2))

        self.norm = nn.LayerNorm(hidden_nf)

        # self.coord_w = nn.Sequential(
        #     nn.Linear(hidden_nf, hidden_nf),
        #     act_fn,
        #     nn.Linear(hidden_nf, 11))

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf * 4),
            act_fn,
            nn.Linear(hidden_nf * 4, 44))
        
        # self.coord_mlp_v = nn.Sequential(
        #     nn.Linear(hidden_nf, hidden_nf),
        #     act_fn,
        #     nn.Linear(hidden_nf, 2))

    def forward(self, h, init_u, init_v, radial, edges, edge_attr, timestamp=None):
        # h = torch.cat([h, self.node_embedding.weight.t()], dim=1)
        h = self.embedding(h) 
        # h = self.embedding1(h[:, :10]) + self.embedding2(h[:, 10:20]) + self.embedding3(h[:, 20:30]) + self.embedding(h[:, 30:31]) + self.embedding4(h[:, 31:41]) + self.embedding5(h[:, 41:51]) + self.embedding5(h[:, 51:61])
        # edge_feat = self.edge_embedding(edge_attr)
        h, u, v, _ = self._modules["gcl_%d" % 0](h, init_u, init_v, radial, edges, edge_attr=edge_attr)
        for i in range(1, self.n_layers):
            # h = self.norm(h)
            h, u, v, _ = self._modules["gcl_%d" % i](h, u, v, radial, edges, edge_attr=edge_attr)

        # speed = torch.sqrt(init_u**2 + init_v**2)
        # dirt = torch.atan2(init_u, init_v)
        # coord_w = self.coord_w(torch.cat([h, speed, dirt], dim=1))
        # u = u + coord_w[:, :11] * init_u
        # v = v + coord_w[:, 11:] * init_v
        
        # decode = self.coord_mlp(h)
        # u = u.unsqueeze(1) * (decode[:, :2].unsqueeze(-1)) 
        # v = v.unsqueeze(1) * (decode[:, 2:].unsqueeze(-1)) 

        decode = self.coord_mlp(h)
        u = u.unsqueeze(1) * (decode[:, :22].reshape(u.size(0), 2, 11)) 
        v = v.unsqueeze(1) * (decode[:, 22:].reshape(v.size(0), 2, 11)) 

        h = self.decoder(h)
        h = h.reshape(h.size(0), self.pred_len, self.output_dim)
        u = u.reshape(u.size(0), self.pred_len, 11)
        v = v.reshape(v.size(0), self.pred_len, 11)
        preds = torch.cat((h[:, :, :30], u[:, :, :10], v[:, :, :10], h[:, :, 30:], u[:, :, 10:], v[:, :, 10:]), dim=-1)
        return preds


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)