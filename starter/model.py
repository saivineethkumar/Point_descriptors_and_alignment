import torch
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm
from torchsummary import summary
import torch.nn.functional as F

# the "MLP" block that you will use the in the PointNet and CorrNet modules you will implement
# This block is made of a linear transformation (FC layer), 
# followed by a Leaky RelU, a Group Normalization (optional, depending on enable_group_norm)
# the Group Normalization (see Wu and He, "Group Normalization", ECCV 2018) creates groups of 32 channels
def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)    
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
                     for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])


# PointNet module for extracting point descriptors
# num_input_features: number of input raw per-point or per-vertex features 
# 		 			  (should be 3, since we have 3D point positions in this assignment)
# num_output_features: number of output per-point descriptors (should be 32 for this assignment)
# this module should include
# - a MLP that processes each point i into a 128-dimensional vector f_i
# - another MLP that further processes these 128-dimensional vectors into h_i (same number of dimensions)
# - a max-pooling layer that collapses all point features h_i into a global shape representaton g
# - a concat operation that concatenates (f_i, g) to create a new per-point descriptor that stores local+global information
# - A MLP followed by a linear transformation layer that transform this concatenated descriptor into the output 32-dimensional descriptor x_i
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class PointNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(PointNet, self).__init__()
        self.f_mlp1 = MLP([num_input_features, 32, 64, 128])
        self.h_mlp2 = MLP([128, 128])
        # self.g_maxpooling = True  #max pool to 128 dim vector
        # self.g_reconstructed = True
        self.y_mlp3 = MLP([256, 128, 64])
        self.y_linear = Lin(64, num_output_features)

    def forward(self, x):
        f = self.f_mlp1(x)
        h = self.h_mlp2(f)
        # g = self.g_maxpooling(h)
        # g = self.g_reconstructed(g)
        g,_ = torch.max(h, dim=0)
        g = torch.unsqueeze(g, dim=0)
        g = g.repeat(f.shape[0], 1)
        fg = torch.cat((f,g), dim=1)
        y = self.y_mlp3(fg)
        y = self.y_linear(y)
        return y


# CorrNet module that serves 2 purposes:  
# (a) uses the PointNet module to extract the per-point descriptors of the point cloud (out_pts)
#     and the same PointNet module to extract the per-vertex descriptors of the mesh (out_vtx)
# (b) if self.train_corrmask=1, it outputs a correspondence mask
# The CorrNet module should
# - include a (shared) PointNet to extract the per-point and per-vertex descriptors 
# - normalize these descriptors to have length one
# - when train_corrmask=1, it should include a MLP that outputs a confidence 
#   that represents whether the mesh vertex i has a correspondence or not
#   Specifically, you should use the cosine similarity to compute a similarity matrix NxM where
#   N is the number of mesh vertices, M is the number of points in the point cloud
#   Each entry encodes the similarity of vertex i with point j
#   Use the similarity matrix to find for each mesh vertex i, its most similar point n[i] in the point cloud 
#   Form a descriptor matrix X = NxF whose each row stores the point descriptor of n[i] (from the point cloud descriptors)
#   Form a vector S = Nx1 whose each entry stores the similarity of the pair (i, n[i])
#   From the PointNet, you also have the descriptor matrix Y = NxF storing the per-vertex descriptors
#   Concatenate [X Y S] into a N x (2F + 1) matrix
#   Transform this matrix into the correspondence mask Nx1 through a MLP followed by a linear transformation
# **** YOU SHOULD CHANGE THIS MODULE, CURRENTLY IT IS INCORRECT ****
class CorrNet(torch.nn.Module):
    def __init__(self, num_output_features, train_corrmask):        
        super(CorrNet, self).__init__()
        self.train_corrmask = train_corrmask
        self.pointnet_share = PointNet(3, num_output_features)
        
        self.mlp = MLP([2*num_output_features +1, 64])
        self.linear = Lin(64, 1)

    def forward(self, vtx, pts):
        out_vtx = self.pointnet_share(vtx)
        out_pts = self.pointnet_share(pts)
        norms = torch.norm(out_vtx, dim=1, keepdim=True)
        out_vtx = out_vtx/norms
        norms = torch.norm(out_pts, dim=1, keepdim=True)
        out_pts = out_pts/norms
        
        if self.train_corrmask:
            Y = out_vtx
            sMat = F.cosine_similarity(out_vtx.unsqueeze(1), out_pts.unsqueeze(0), dim=-1)
            S, smaxIndices = torch.max(sMat, dim=1, keepdim=True)
            smaxIndices = torch.squeeze(smaxIndices)
            X = out_pts.index_select(dim=0, index=smaxIndices)

            YXS = torch.cat((Y,X,S), dim=1)
            out_corrmask = self.mlp(YXS)
            out_corrmask = self.linear(out_corrmask)

        else:
            out_corrmask = None

        return out_vtx, out_pts, out_corrmask



# Testing model 
# model = PointNet(3, 32)
# print(model)
# input_tensor = torch.randn(11, 3)
# output_tensor = model(input_tensor)
# print(output_tensor.shape)

# model = CorrNet(32, True)
# print(model)
# v_input = torch.randn(100, 3)
# p_input = torch.randn(120, 3)
# a,b,c = model(v_input, p_input)
# print(a.shape, b.shape, c.shape)