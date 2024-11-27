import torch
from torch_geometric.data import Data
import logging
from .backbone import *
from .decoder import *
from lightconvpoint.spatial import knn, sampling_quantized as sampling
from lightconvpoint.utils.functional import batch_gather
from lightconvpoint.nn import max_pool, interpolate

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.data import Data as gData

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GATOccupancyPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=4, out_channels=2):
        """
        Initialize the GAT-based occupancy predictor.
        :param in_channels: Number of input features per node.
        :param hidden_channels: Number of hidden units for GATConv.
        :param num_heads: Number of attention heads in GATConv.
        :param out_channels: Number of output features per node (default: 2 for occupancy classification).
        """
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.fc = nn.Linear(hidden_channels * num_heads, out_channels)
        self.out_channels = out_channels

    def forward(self, data, spatial_only=False, spectral_only=False):
        if spatial_only:
            return data
        pos = data["pos"]
        pos_non_manifold = data["pos_non_manifold"]

        """
        Forward pass for the GAT-based model.
        :param pos: Tensor of surface points [N, 3].
        :param pos_non_manifold: Tensor of points in bounding volume [M, 3].
        :param batch: Batch information for graph processing.
        :return: Occupancy predictions of shape [Batch size, 2, M].
        """

        # Combine surface points and bounding volume points
        all_points = torch.cat([pos, pos_non_manifold], dim=2)
        all_points = all_points.transpose(1,2)

        batch_size, num_nodes, fc_dim = all_points.shape
                # Compute adjacency matrix (dense to sparse format for PyG)
        dist_matrix = torch.cdist(all_points, all_points)  # Pairwise Euclidean distances
        # print(dist_matrix.mean(dim=1), dist_matrix.max())
        adjacency_matrix = ( dist_matrix < 0.05)# Threshold-based connectivity

        batch_adj = torch.block_diag(*[adj for adj in adjacency_matrix])
        # Flatten node features into a single tensor
        batch_nodes = all_points.reshape(-1, fc_dim)  # Shape: (batch_size * num_nodes, fc_dim)

        # Create a batch index tensor mapping nodes to their respective graphs
        batch_vector = torch.repeat_interleave(torch.arange(batch_size), num_nodes)

        # Create PyTorch Geometric Data
        data = Data(x=batch_nodes, edge_index=batch_adj.nonzero(as_tuple=False).T, batch=batch_vector)

        # Output for verification
        # print("Node Features Shape:", data.x.shape)       # (batch_size * num_nodes, fc_dim)
        # print("Edge Index Shape:", data.edge_index.shape) # (2, num_edges)
        # print("Batch Shape:", data.batch.shape)           # (batch_size * num_nodes,)

        x = F.relu(self.gat1(data.x, data.edge_index))
        x = F.relu(self.gat2(x, data.edge_index))
        x = self.fc(x)
        x =  x.view(batch_size, num_nodes, self.out_channels)
        # print("out shape",x.shape)




        # adjacency_matrix = dist_matrix
        del dist_matrix
        edge_index, edge_attr = dense_to_sparse(adjacency_matrix)
        # print("adjacency_matrix",adjacency_matrix.shape)
        # edge_index = adjacency_matrix.nonzero().t().contiguous()
        # edge_index_list = self.build_edge_index(batch_size, num_nodes)
        # all_points = self.build_graph_batch(all_points, edge_index)
# 
        # print("all_points.x", all_points.shape, edge_index.shape)

        # GAT layers
        # x = F.relu(self.gat1(all_points[0], edge_index))
        # x = F.relu(self.gat2(x, edge_index))
        # x = self.fc(x)


        # Extract outputs for non-manifold points (M points)
        # batch_size = batch.max().item() + 1
        batch_size= pos.shape[0]
        num_points = pos_non_manifold.size(2)
      
        # occupancy_logits = x[-num_points:].reshape(batch_size, self.out_channels, num_points)  # [Batch size,out_channels, M]
        # occupancy_logits = x[num_points:].reshape(batch_size, self.out_channels, 3000)  # [Batch size,out_channels, M]
        
        occupancy_logits = x[:,num_points:].reshape(batch_size, self.out_channels, 3000)  # [Batch size,out_channels, M]

        return occupancy_logits
    
    # def build_graph_batch(self, x_list,  edge_index_list):
    #     data_list = []
    #     for graph_nodes, ei in zip(x_list, edge_index_list):
    #         data_list.append(gData(x=graph_nodes, edge_index=ei))
    #     graph = Batch.from_data_list(data_list)
    #     return graph
    # def build_edge_index(self,  batch_size, num_nodes):
    #     row = torch.arange(num_nodes).view(-1, 1).repeat(1, num_nodes).view(-1)
    #     col = torch.arange(num_nodes).view(-1, 1).repeat(num_nodes, 1).view(-1)
    #     edge_index = torch.stack([row, col], dim=0)
    #     edge_index_list = [edge_index for _ in range(batch_size)]
    #     return edge_index_list

class Network(torch.nn.Module):

    def __init__(self, in_channels, latent_size, out_channels, backbone, decoder, **kwargs):
        super().__init__()

        # self.net = eval(backbone)(in_channels, latent_size, segmentation=True, dropout=0)
        self.net = GATOccupancyPredictor(3, latent_size, num_heads=4, out_channels=latent_size)
        self.projection = eval(decoder["name"])(latent_size, out_channels, decoder["k"])
        self.lcp_preprocess = True

        logging.info(f"Network -- backbone -- {count_parameters(self.net)} parameters")
        logging.info(f"Network -- projection -- {count_parameters(self.projection)} parameters")

    def forward(self, data, spatial_only=False, spectral_only=False):

        if spatial_only:
            net_data = self.net(data, spatial_only=spatial_only)
            if "output_support" in net_data:
                data["output_support"] = net_data["output_support"]
            proj_data = self.projection.forward_spatial(data)
            net_data["proj_indices"] = proj_data["proj_indices"]
            return net_data

        if not spectral_only:
            spatial_data = self.net.forward_spatial(data)
            if "output_support" in spatial_data:
                data["output_support"] = spatial_data["output_support"]
            proj_data = self.projection.forward_spatial(data)
            spatial_data["proj_indices"] = proj_data["proj_indices"]
            for key, value in spatial_data.items():
                data[key] = value

        latents = self.net(data, spectral_only=True)
        print("latents", latents.shape)
        data["latents"] = latents
        ret_data = self.projection(data, spectral_only=True)

        return ret_data



    def get_latent(self, data, with_correction=False, spatial_only=False, spectral_only=False):

        latents = self.net(data, spatial_only=spatial_only, spectral_only=spectral_only)
        data["latents"] = latents

        data["proj_correction"] = None
        if with_correction:
            data_in_proj = {"latents":latents, "pos":data["pos"], "pos_non_manifold":data["pos"].clone(), "proj_correction":None}
            data_proj = self.projection(data_in_proj, spectral_only=False)
            data["proj_correction"] = data_proj
        return data

    def from_latent(self, data):
        data_proj = self.projection(data)
        return data_proj#["outputs"]


class NetworkMultiScale(torch.nn.Module):

    def __init__(self, in_channels, latent_size, out_channels, backbone, decoder, **kwargs):
        super().__init__()

        self.net = eval(backbone)(in_channels, latent_size, segmentation=True, dropout=0)

        self.merge_latent = torch.nn.Sequential(
            torch.nn.Conv1d(2*latent_size, latent_size,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(latent_size, latent_size,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(latent_size, latent_size,1)
        )

        if "Radius" in decoder["name"]:
            self.projection = eval(decoder["name"])(latent_size, out_channels, decoder["radius"])
        else:
            self.projection = eval(decoder["name"])(latent_size, out_channels, decoder["k"])
        self.lcp_preprocess = True

        logging.info(f"Network -- backbone -- {count_parameters(self.net)} parameters")
        logging.info(f"Network -- projection -- {count_parameters(self.projection)} parameters")

    def forward(self, data):

        # compute the down sampled latents
        # ids_down = torch.rand((data["pos"].shape[0], 3000), device=data["pos"].device) * data["pos"].shape[2]
        # ids_down = ids_down.long()

        with torch.no_grad():
            pos_down, idsDown = sampling(data["pos"], n_support=3000)
            x_down = batch_gather(data["x"], dim=2, index=idsDown).contiguous()
            data_down = {'x':x_down, 'pos':pos_down}
            latents_down = self.net(data_down)
            idsUp = knn(pos_down, data["pos"], 1)
            latents_down = interpolate(latents_down, idsUp)
        
        latents = self.net(data)
        
        latents = torch.cat([latents, latents_down], dim=1)
        latents = self.merge_latent(latents)

        data["latents"] = latents
        ret_data = self.projection(data)

        return ret_data

    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        # set only the merge to train
        for module in self.children():
            module.train(False)
        self.merge_latent.train(mode)
        return self

    def get_latent(self, data, with_correction=False, spatial_only=False, spectral_only=False):

        latents = self.net(data, spatial_only=spatial_only, spectral_only=spectral_only)
        data["latents"] = latents

        data["proj_correction"] = None
        if with_correction:
            data_in_proj = {"latents":latents, "pos":data["pos"], "pos_non_manifold":data["pos"].clone(), "proj_correction":None}
            data_proj = self.projection(data_in_proj, spectral_only=False)
            data["proj_correction"] = data_proj
        return data

    def from_latent(self, data):
        data_proj = self.projection(data)
        return data_proj#["outputs"]
