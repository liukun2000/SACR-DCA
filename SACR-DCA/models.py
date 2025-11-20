import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.graphgym import GATConv
# GCN layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
          
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):

        support = torch.mm(x, self.weight)

        output = torch.sparse.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

# GCN model
class GraphGCN(nn.Module):
    def __init__(self, n_input, n_gcn_1, dim_final, dropout):
        super(GraphGCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(n_input, n_gcn_1)        #GCN1
        self.gc2 = GraphConvolution(n_gcn_1 , dim_final)   #GCN2

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z
class AE_encoder(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.z_layer = Linear(ae_n_enc_2, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z_ae = self.z_layer(z)
        return z_ae
class AE_decoder(nn.Module):
    def __init__(self, ae_n_dec_1, ae_n_dec_2, n_input, n_z):
        super(AE_decoder, self).__init__()
        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.x_bar_layer = Linear(ae_n_dec_2, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


class AttentionFusionModule(nn.Module):
    def __init__(self, feature_dim, feature_num):
        super(AttentionFusionModule, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(feature_num * feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_num),
            nn.Softmax(dim=-1)
        )
        self.feature_num = feature_num
    def forward(self, z: list):
        combined = torch.cat(z, dim=1)
        attention_weights = self.attention_network(combined)
        weighted_z = []
        for i in range(0, self.feature_num):
            weighted_z_i = attention_weights[:, i:i + 1] * z[i]
            weighted_z.append(weighted_z_i)

        fused_z = sum(weighted_z)

        return fused_z
###############Model 1#######################
class Model1(nn.Module):
    def __init__(self, omic_dims,ae_n_enc_1,ae_n_enc_2 ,n_gcn_1, dim_final, dropout):
        super(Model1, self).__init__()
        self.dropout = dropout
        self.ae_encoders = nn.ModuleDict()
        self.gcns = nn.ModuleDict()
        for omic, dim in omic_dims.items():
            self.ae_encoders[omic] = AE_encoder(
                ae_n_enc_1=ae_n_enc_1,
                ae_n_enc_2=ae_n_enc_2,
                n_input=dim,
                n_z=dim_final
            )
            self.gcns[omic] = GraphGCN(
                n_input=dim,
                n_gcn_1=n_gcn_1,
                dim_final=dim_final,
                dropout=dropout
            )
        self.attention_fuse=AttentionFusionModule(feature_dim=dim_final,feature_num=len(omic_dims)*2)
    def forward(self, omic_data, adj_matrices):
        ae_representations = []
        gcn_representations = []
        for omic in omic_data.keys():
            ae_representations.append(self.ae_encoders[omic](omic_data[omic]))
            gcn_representations.append(self.gcns[omic](omic_data[omic], adj_matrices[omic]))

        z_fused = self.attention_fuse(ae_representations+gcn_representations)

        return z_fused,ae_representations,gcn_representations

###############Model 2#######################
class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_channels, out_channels*2)
        self.gc2 = GraphConvolution(out_channels*2 , out_channels)

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z
class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_channels, in_channels*2)
        self.gc2 = GraphConvolution(in_channels*2 , out_channels)

    def forward(self, x, adj):
        z = self.gc1(x, adj)
        z = F.leaky_relu(z, 0.25)
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.gc2(z, adj)
        z = F.leaky_relu(z, 0.25)
        return z

class Model2(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, num_sample: int,input_dim,output_dim):
        super(Model2, self).__init__()
        self.n = num_sample
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(self.n, self.n, dtype=torch.float32), requires_grad=True)
        self.linear = nn.Linear(input_dim, output_dim)


    def forward(self, x, edge_index):
        H = self.encoder(x, edge_index) 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        C_diag = torch.diag(torch.diag(self.Coefficient)).to(device)
        Coefficient = self.Coefficient
        CH = torch.matmul(Coefficient, H)

        X_ = self.decoder(CH, edge_index)
        y = self.linear(CH)
        y1=F.softmax(y,dim=1)
        W = 0.5 * (Coefficient + Coefficient.T)
        return H, CH, Coefficient, X_,y1,W

