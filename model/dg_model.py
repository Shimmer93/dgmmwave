import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from pyskl.models.gcns.ctrgcn import CTRGCNBlock
# from misc.skeleton import MMWaveGraph

class JointAttention(nn.Module):
    def __init__(self, num_joints=13, dim=64):
        super(JointAttention, self).__init__()
        self.num_joints = num_joints
        self.dim = dim
        self.joint_emb = nn.Parameter(torch.randn(1, num_joints, dim))

    def forward(self, x):
        B, N, D = x.shape

        joint_emb = self.joint_emb.expand(B, -1, -1) # [B, J, D]
        attn = x @ joint_emb.transpose(1, 2) # [B, N, J]
        attn = F.softmax(attn, dim=1) # [B, N, J]
        x = attn.transpose(1, 2) @ x # [B, J, D]

        return x
    
class CTRGCN(nn.Module):
    def __init__(self, 
                 graph_cfg,
                 base_channels=64,
                 num_stages=10,
                 pretrained=None,
                 **kwargs):
        super(CTRGCN, self).__init__()

        self.graph = MMWaveGraph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.base_channels = base_channels
        self.data_bn = nn.BatchNorm1d(base_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        modules = []
        for i in range(num_stages):
            kwargs_i = kwargs0 if i == 0 else kwargs
            modules.append(CTRGCNBlock(base_channels, base_channels, A.clone(), **kwargs_i))
        self.net = nn.ModuleList(modules)

    def forward(self, x):
        B, T, J, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, J*C, T) # [B, J*C, T]
        x = self.data_bn(x)
        x = x.reshape(B, J, C, T).permute(0, 2, 3, 1) # [B, C, T, J]

        for layer in self.net:
            x = layer(x)

        x = x.permute(0, 2, 3, 1) # [B, T, J, C]
        return x

class DGModel(nn.Module):
    def __init__(self,
                 graph_layout,
                 graph_mode='spatial',
                 num_features=2,
                 num_joints=13,
                 num_layers_point=6,
                 num_layers_joint=6,
                 dim=64,
                 num_heads=8,
                 dim_feedforward=256,
                 dropout=0.1):
        super(DGModel, self).__init__()

        self.pos_enc = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.feat_enc = nn.Sequential(
            nn.Linear(num_features, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

        self.point_mixer = TransformerEncoder(
            TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers_point
        )

        self.joint_attn = JointAttention(num_joints=num_joints, dim=dim)
        self.joint_ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.joint_gcn = CTRGCN(dict(layout=graph_layout, mode=graph_mode), base_channels=dim, num_stages=num_layers_joint)
        
        self.pos_dec = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 3)
        )

    def pos_autoencode(self, coords):
        pos_embs = self.pos_enc(coords)
        rec_coords = self.pos_dec(pos_embs)
        return pos_embs, rec_coords
    
    def feat_autoencode(self, feats):
        feat_embs = self.feat_enc(feats)
        rec_feats = self.feat_dec(feat_embs)
        return feat_embs, rec_feats

    def forward_train(self, pcs, gt_skls):
        # pcs: [B, T, N, 3+num_features]
        # gt_skls: [B, J, 3]

        B, T, N, _ = pcs.shape
        _, J, _ = gt_skls.shape

        # separate coordinates and features
        coords, feats = pcs[:,:,:,:3], pcs[:,:,:,3:]

        # autoencode point clouds and ground truth skeletons
        pos_embs_pc, rec_pc = self.pos_autoencode(coords)
        pos_embs_skl, rec_skl = self.pos_autoencode(gt_skls)
        l_rec_pc = F.mse_loss(rec_pc, coords)
        l_rec_skl = F.mse_loss(rec_skl, gt_skls)

        # encode features
        feat_embs = self.feat_enc(feats)
        x = pos_embs_pc + feat_embs
        x = x.reshape(B*T, N, -1)
        
        # transformer encoder for points
        x = self.point_mixer(x)
        x = self.joint_attn(x)
        x = self.joint_ff(x)
        x = x.reshape(B, T, J, -1)


        # graph convolution for joints
        x = self.joint_gcn(x)
        x = torch.mean(x, dim=1)
        l_pos = F.mse_loss(x, pos_embs_skl)

        with torch.no_grad():
            skl = self.pos_dec(x)

        return l_rec_pc, l_rec_skl, l_pos, skl


    def forward(self, input): # [B, T, N, 3]
        B, T, N, _ = input.shape

        # separate coordinates and features
        coords, feats = input[:,:,:,:3], input[:,:,:,3:] # [B, T, N, 3], [B, T, N, C]
        
        # encode coordinates and features
        pos_embs = self.pos_enc(coords) # [B, T, N, D]
        feat_embs = self.feat_enc(feats) # [B, T, N, D]
        x = pos_embs + feat_embs
        x = x.reshape(B*T, N, -1)

        # transformer encoder for points
        x = self.point_mixer(x) # [B*T, N, D]
        x = self.joint_attn(x) # [B*T, J, D]
        x = self.joint_ff(x) # [B*T, J, D]
        _, J, _ = x.shape
        x = x.reshape(B, T, J, -1)

        # graph convolution for joints
        x = self.joint_gcn(x) # [B, T, J, D]
        x = torch.mean(x, dim=1) # [B, J, D]

        # decode to skeleton
        skl = self.pos_dec(x) # [B, J, 3]

        return skl