import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn.conv import MessagePassing

def _rbf(D, D_min=0., D_max=6., D_count=9, device='cpu'):
    
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    
    return RBF


def gnn_norm(x, norm):

    batch_size, num_nodes, num_channels = x.size()
    x = x.view(-1, num_channels)
    x = norm(x)
    x = x.view(batch_size, num_nodes, num_channels)

    return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(drop_rate),
        )
        
    def forward(self, x):
        
        return self.mlp(x)
    
class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, drop_rate, output_dim):
        super(FC, self).__init__()
        
        self.predict = nn.ModuleList()
        self.predict.append(MLP(input_dim, hidden_dim, drop_rate))
        for _ in range(num_layer - 2):
            self.predict.append(MLP(hidden_dim, hidden_dim, drop_rate))
        self.predict.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, h):
        
        for layer in self.predict:
            h = layer(h)
            
        return h

class GeometryAwareLayer(MessagePassing):
    def __init__(self, input_dim, output_dim, drop_rate, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GeometryAwareLayer, self).__init__(**kwargs)
        
        self.mlp_coord = MLP(9, input_dim, 0.0)
        self.out = MLP(input_dim, output_dim, drop_rate)
        
    def message(self, x_j, x_i, radial, index):
        
        return x_j * radial
    
    def forward(self, x, data, edge_index):
        
        res = x

        pos, size = data.pos, None
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        dist = torch.norm(coord_diff, p=2, dim=-1)
        radial = self.mlp_coord(_rbf(dist, device=x.device))
        x = self.propagate(edge_index=edge_index, x=x, radial=radial, size=size)
        
        x = self.out(x) + res

        return x

class GeometryAwareBlock(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(GeometryAwareBlock, self).__init__()
        
        self.gconv_intra = GeometryAwareLayer(input_dim, output_dim, drop_rate)
        self.gconv_inter = GeometryAwareLayer(input_dim, output_dim, drop_rate)

    def forward(self, x, data):
        
        x_intra = self.gconv_intra(x, data, data.edge_index_intra)
        x_inter = self.gconv_inter(x, data, data.edge_index_inter)
        x = (x_intra + x_inter) / 2

        return x

class DiffPool(nn.Module):
    def __init__(self, input_dim, output_dim, max_num, red_node, edge, drop_rate):
        super().__init__()

        self.max_num = max_num
        self.red_node = red_node
        self.edge = edge
        self.gnn_p = DenseGCNConv(input_dim, red_node, improved=True, bias=True)
        self.gnn_p_norm = nn.Sequential(
            nn.BatchNorm1d(red_node),
            nn.Mish(),
        )
        self.gnn_e = DenseGCNConv(input_dim, output_dim, improved=True, bias=True)
        self.gnn_e_norm = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
        )
        self.out = nn.Linear(output_dim, output_dim)
        self.out_norm = nn.Sequential(
            nn.BatchNorm1d(output_dim),
        )

    def pooling(self, x, adj, s, mask=None):

        batch_size, num_nodes, _ = x.size()
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        s = F.softmax(s, dim=-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        return out, out_adj, s
    
    def set_edge_index(self, data, edge):

        switch = {
            "intra": data.edge_index_intra,
            "inter": data.edge_index_inter,
            "intra_lig": data.edge_index_intra_lig,
            "intra_pro": data.edge_index_intra_pro,
        }
        data.edge_index = switch.get(edge, None)

    def forward(self, x, data):



        if self.edge == "intra_lig":


            x = x
            edge_index_lig = data.edge_index_intra_lig
            batch = data.batch
            split = data.split


            lig_mask = (split == 0)


            x_alone = x[lig_mask]
            batch_alone = batch[lig_mask]


            device = x.device
            num_nodes = x.size(0)

            new_index = -torch.ones(num_nodes, dtype=torch.long, device=device)
            new_index[lig_mask] = torch.arange(lig_mask.sum(),
                                               device=device)


            edge_index_alone = new_index[edge_index_lig]

        elif self.edge == "intra_pro":


            x = x
            edge_index_pro = data.edge_index_intra_pro
            batch = data.batch
            split = data.split


            pro_mask = (split == 1)


            x_alone = x[pro_mask]
            batch_alone = batch[pro_mask]


            device = x.device
            num_nodes = x.size(0)

            new_index = -torch.ones(num_nodes, dtype=torch.long, device=device)
            new_index[pro_mask] = torch.arange(pro_mask.sum(),
                                               device=device)


            edge_index_alone = new_index[edge_index_pro]

        else:
            print("edge ERROR")

        adj = to_dense_adj(edge_index_alone, batch_alone,
                           max_num_nodes=self.max_num)
        x, mask = to_dense_batch(x_alone, batch_alone, fill_value=0,
                                 max_num_nodes=self.max_num)




        s = gnn_norm(self.gnn_p(x, adj, mask),
                     self.gnn_p_norm)
        x, adj, s = self.pooling(x, adj, s,
                                 mask)
        x = gnn_norm(self.gnn_e(x, adj), self.gnn_e_norm)
        x = gnn_norm(self.out(x), self.out_norm)

        return x, s
    
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, heads, drop_rate):
        super().__init__()

        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = MLP(hidden_dim, hidden_dim, drop_rate)
        
    def forward(self, q, k, v): 

        res = q.sum(dim=1)

        batch_size, seqlen_q, _ = q.shape
        _, seqlen_k, _ = k.shape
        
        Q = self.W_Q(q)
        K = self.W_K(k)
        V = self.W_V(v)
        
        Q = Q.view(batch_size, seqlen_q, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seqlen_k, self.heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, seqlen_q, self.hidden_dim)
        x = x.sum(dim=1)

        x = self.W_O(x) + res
        
        return x, attention

class Model(nn.Module):
    def __init__(self,
                 K_modes_ligand, K_u_ligand,
                 K_modes_protein, K_u_protein,
                 K_modes_complex, K_u_complex,
                 node_dim, hidden_dim, num_clusters=[28, 156], heads=1, drop_rate=0.1):
        super().__init__()
        
        self.embedding = MLP(node_dim, hidden_dim, 0.0)
        self.GeometryAwareBlock1 = GeometryAwareBlock(hidden_dim, hidden_dim, drop_rate)
        self.GeometryAwareBlock2 = GeometryAwareBlock(hidden_dim, hidden_dim, drop_rate)
        self.GeometryAwareBlock3 = GeometryAwareBlock(hidden_dim, hidden_dim, drop_rate)
        self.diffpool1 = DiffPool(hidden_dim, hidden_dim, 600, num_clusters[0], "intra_lig", drop_rate)
        self.diffpool2 = DiffPool(hidden_dim, hidden_dim, 600, num_clusters[1], "intra_pro", drop_rate)
        self.attblock1 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.attblock2 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.fc = FC(hidden_dim, hidden_dim, 2, drop_rate, 1)


        self.hidden_dim = hidden_dim
        self.K_nums = {
            "K_modes_ligand": K_modes_ligand,
            "K_u_ligand": K_u_ligand,
            "K_modes_protein": K_modes_protein,
            "K_u_protein": K_u_protein,
            "K_modes_complex": K_modes_complex,
            "K_u_complex": K_u_complex
            }
        
        self.embedding_u_ligand = MLP_u(1, hidden_dim, 0.0)
        self.embedding_u_protein = MLP_u(1, hidden_dim, 0.0)
        self.embedding_u_complex = MLP_u(1, hidden_dim, 0.0)

        self.attblock_u_1 = AttentionBlock(hidden_dim, heads, drop_rate)
        self.attblock_u_2 = AttentionBlock(hidden_dim, heads, drop_rate)



    def make_edge_index(self, data):

        data.edge_index_intra_lig = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 0]
        data.edge_index_intra_pro = data.edge_index_intra[:, data.split[data.edge_index_intra[0, :]] == 1]

    def forward(self, batch):
        
        batch_complex, batch_anm_ligand, batch_anm_protein, batch_anm_complex = batch



        Km_l = self.K_nums["K_modes_ligand"]
        Ku_l = self.K_nums["K_u_ligand"]

        batch_anm_ligand.u = batch_anm_ligand.u[:, :Ku_l]


        Km_p = self.K_nums["K_modes_protein"]
        Ku_p = self.K_nums["K_u_protein"]

        batch_anm_protein.u = batch_anm_protein.u[:, :Ku_p]


        Km_c = self.K_nums["K_modes_complex"]
        Ku_c = self.K_nums["K_u_complex"]

        batch_anm_complex.u = batch_anm_complex.u[:, :Ku_c]

        

        data = batch_complex


        x = data.x
        x = self.embedding(x)


        self.make_edge_index(data)
        x = self.GeometryAwareBlock1(x, data)
        x = self.GeometryAwareBlock2(x, data)
        x = self.GeometryAwareBlock3(x, data)


        x_lig, _ = self.diffpool1(x, data)
        x_pro, _  = self.diffpool2(x, data)

        l2p, _ = self.attblock1(x_lig, x_pro, x_pro)
        p2l, _ = self.attblock2(x_pro, x_lig, x_lig)
        x = l2p + p2l



        B, K_ul = batch_anm_ligand.u.shape
        _, K_up = batch_anm_protein.u.shape
        _, K_uc = batch_anm_complex.u.shape

        hidden_dim = self.hidden_dim


        u_l = self.embedding_u_ligand(batch_anm_ligand.u.unsqueeze(-1))


        u_p = self.embedding_u_protein(batch_anm_protein.u.unsqueeze(-1))


        u_c = self.embedding_u_complex(batch_anm_complex.u.unsqueeze(-1))


        assert u_l.shape == (B, K_ul, hidden_dim)
        assert u_p.shape == (B, K_up, hidden_dim)
        assert u_c.shape == (B, K_uc, hidden_dim)


        l2c_u, _ = self.attblock_u_1(u_l, u_c, u_c)
        p2c_u, _ = self.attblock_u_2(u_p, u_c, u_c)
        x = x + l2c_u + p2c_u



        x = self.fc(x)

        return x.view(-1)
    


class MLP_u(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.Mish()
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):

        x = self.fc(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x
