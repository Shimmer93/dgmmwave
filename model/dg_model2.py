import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# try:
#     from pyskl.models.gcns.ctrgcn import CTRGCNBlock
# except ImportError:
#     print("Please install the package 'pyskl' to use dg_model.py.")
# from misc.skeleton import MMWaveGraph

class JointAttention(nn.Module):
    def __init__(self, num_joints=13, dim=64):
        super(JointAttention, self).__init__()
        self.num_joints = num_joints
        self.dim = dim
        self.joint_emb = nn.Parameter(torch.randn(1, num_joints, dim))
        self.to_kv = nn.Linear(dim, dim*2)

    def forward(self, x):
        B, N, D = x.shape

        k, v = self.to_kv(x).chunk(2, dim=-1)

        joint_emb = self.joint_emb.expand(B, -1, -1) # [B, J, D]
        attn = k @ joint_emb.transpose(1, 2) # [B, N, J]
        attn = F.softmax(attn, dim=1) # [B, N, J]
        x = attn.transpose(1, 2) @ v # [B, J, D]

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

class DGModel2(nn.Module):
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
        super(DGModel2, self).__init__()

        self.aux_points = nn.Parameter(torch.randn(1, 1, 64, 3+num_features))

        self.pos_enc = nn.Sequential(
            nn.Linear(3, dim//4),
            nn.LayerNorm(dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, dim//2)
        )

        self.feat_enc = nn.Sequential(
            nn.Linear(num_features, dim//4),
            nn.LayerNorm(dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, dim//2)
        )

        self.point_mixer = TransformerEncoder(
            TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers_point
        )

        self.joint_attn = JointAttention(num_joints=num_joints, dim=dim)
        self.joint_ff = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.LayerNorm(2*dim),
            nn.ReLU(),
            nn.Linear(2*dim, dim)
        )
        self.joint_gcn = CTRGCN(dict(layout=graph_layout, mode=graph_mode), base_channels=dim, num_stages=num_layers_joint)
        
        self.pos_dec = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1024+64)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def pos_autoencode(self, coords):
        pos_embs = self.pos_enc(coords)
        rec_coords = self.pos_dec(pos_embs)
        return pos_embs, rec_coords
    
    def feat_autoencode(self, feats):
        feat_embs = self.feat_enc(feats)
        rec_feats = self.feat_dec(feat_embs)
        return feat_embs, rec_feats

    def forward_train(self, pcs, gt_skls):
        aux_points = self.aux_points.expand(pcs.shape[0], pcs.shape[1], -1, -1)
        pcs = torch.cat([pcs, aux_points], dim=2)
        # pcs: [B, T, N, 3+num_features]
        # gt_skls: [B, J, 3]

        B, T, N, _ = pcs.shape
        _, _, J, _ = gt_skls.shape

        # separate coordinates and features
        coords, feats = pcs[:,:,:,:3], pcs[:,:,:,3:]

        # # autoencode point clouds and ground truth skeletons
        # pos_embs_pc, rec_pc = self.pos_autoencode(coords)
        # pos_embs_skl, rec_skl = self.pos_autoencode(gt_skls)
        # l_rec_pc = F.mse_loss(rec_pc, coords)
        # l_rec_skl = F.mse_loss(rec_skl, gt_skls)
        pos_embs_pc = self.pos_enc(coords)

        # encode features
        feat_embs = self.feat_enc(feats)
        x = torch.cat([pos_embs_pc, feat_embs], dim=-1)
        x = x.reshape(B*T, N, -1)
        
        # transformer encoder for points
        x = self.point_mixer(x)
        x = self.joint_attn(x)
        x = self.joint_ff(x)
        x = x.reshape(B, T, J, -1)

        # graph convolution for joints
        x = self.joint_gcn(x) # [B, T, J, D]
        coord = coords[:,x.shape[1]//2,...]
        x = x[:,x.shape[1]//2,...]

        # decode to skeleton
        x = self.pos_dec(x) # [B, J, N]
        x = F.softmax(x, dim=-1)
        skl = torch.bmm(x, coord).unsqueeze(1) # [B, 1, J, 3]

        l_pos = F.mse_loss(skl, gt_skls)

        return l_pos, skl

    def forward(self, input): # [B, T, N, 3]
        aux_points = self.aux_points.expand(input.shape[0], input.shape[1], -1, -1)
        input = torch.cat([input, aux_points], dim=2)

        B, T, N, _ = input.shape

        # separate coordinates and features
        coords, feats = input[:,:,:,:3], input[:,:,:,3:] # [B, T, N, 3], [B, T, N, C]
        
        # encode coordinates and features
        pos_embs = self.pos_enc(coords) # [B, T, N, D]
        feat_embs = self.feat_enc(feats) # [B, T, N, D]
        # x = pos_embs + feat_embs
        x = torch.cat([pos_embs, feat_embs], dim=-1)
        x = x.reshape(B*T, N, -1)

        # transformer encoder for points
        x = self.point_mixer(x) # [B*T, N, D]
        x = self.joint_attn(x) # [B*T, J, D]
        x = self.joint_ff(x) # [B*T, J, D]
        _, J, _ = x.shape
        x = x.reshape(B, T, J, -1)

        # graph convolution for joints
        x = self.joint_gcn(x) # [B, T, J, D]
        coord = coords[:,x.shape[1]//2,...]
        x = x[:,x.shape[1]//2,...]

        # decode to skeleton
        x = self.pos_dec(x) # [B, J, N]
        x = F.softmax(x, dim=-1)
        skl = torch.bmm(x, coord).unsqueeze(1) # [B, 1, J, 3]

        return skl