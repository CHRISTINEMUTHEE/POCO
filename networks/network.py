import torch
import time
from torch_geometric.data import Data
import logging
from .backbone import *
from .decoder import *
from lightconvpoint.spatial import knn, sampling_quantized as sampling
from lightconvpoint.utils.functional import batch_gather
from lightconvpoint.nn import max_pool, interpolate

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn.pool import knn_graph
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.data import Data as gData

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class GATOccupancyPredictor(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_heads=4, out_channels=2):
#         """
#         Initialize the GAT-based occupancy predictor.
#         :param in_channels: Number of input features per node.
#         :param hidden_channels: Number of hidden units for GATConv.
#         :param num_heads: Number of attention heads in GATConv.
#         :param out_channels: Number of output features per node (default: 2 for occupancy classification).
#         """
#         super().__init__()
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
#         self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
#         self.gat3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
#         # self.gat4 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
#         # self.gat5 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
#         self.fc = nn.Linear(hidden_channels * num_heads, out_channels)
#         self.out_channels = out_channels

#     def forward(self, data, spatial_only=False, spectral_only=False):
#         if spatial_only:
#             return data
#         pos = data["pos"]
#         pos_non_manifold = data["pos_non_manifold"]

#         """
#         Forward pass for the GAT-based model.
#         :param pos: Tensor of surface points [N, 3].
#         :param pos_non_manifold: Tensor of points in bounding volume [M, 3].
#         :param batch: Batch information for graph processing.
#         :return: Occupancy predictions of shape [Batch size, 2, M].
#         """

#         # Combine surface points and bounding volume points
#         all_points = torch.cat([pos, pos_non_manifold], dim=2)
#         all_points = all_points.transpose(1,2)

#         batch_size, num_nodes, fc_dim = all_points.shape
#                 # Compute adjacency matrix (dense to sparse format for PyG)
#         start_time = time.time()
#         print(f"Before computing adjacency matrix: ")
#         dist_matrix = torch.cdist(all_points, all_points)  # Pairwise Euclidean distances
#         print(f"Time taken to compute adjacency matrix: {time.time() - start_time:.2f} s")
#         # print(dist_matrix.mean(dim=1), dist_matrix.max())
#         adjacency_matrix = ( dist_matrix < 0.05)# Threshold-based connectivity

#         batch_adj = torch.block_diag(*[adj for adj in adjacency_matrix])
#         # Flatten node features into a single tensor
#         batch_nodes = all_points.reshape(-1, fc_dim)  # Shape: (batch_size * num_nodes, fc_dim)

#         # Create a batch index tensor mapping nodes to their respective graphs
#         batch_vector = torch.repeat_interleave(torch.arange(batch_size), num_nodes)

#         # Create PyTorch Geometric Data
#         data = Data(x=batch_nodes, edge_index=batch_adj.nonzero(as_tuple=False).T, batch=batch_vector)

#         # Output for verification
#         # print("Node Features Shape:", data.x.shape)       # (batch_size * num_nodes, fc_dim)
#         # print("Edge Index Shape:", data.edge_index.shape) # (2, num_edges)
#         # print("Batch Shape:", data.batch.shape)           # (batch_size * num_nodes,)

#         # x = F.relu(self.gat1(data.x, data.edge_index))
#         # x = F.relu(self.gat2(x, data.edge_index))
#         x = F.relu(self.gat1(data.x, data.edge_index))
#         x = F.relu(self.gat2(x, data.edge_index))
#         x = F.relu(self.gat3(x, data.edge_index))
#         # x = F.relu(self.gat4(x, data.edge_index))
#         # x = F.relu(self.gat5(x, data.edge_index))
#         x = self.fc(x)
#         x =  x.view(batch_size, num_nodes, self.out_channels)
#         # print("out shape",x.shape)




#         # adjacency_matrix = dist_matrix
#         del dist_matrix
#         edge_index, edge_attr = dense_to_sparse(adjacency_matrix)
#         # print("adjacency_matrix",adjacency_matrix.shape)
#         # edge_index = adjacency_matrix.nonzero().t().contiguous()
#         # edge_index_list = self.build_edge_index(batch_size, num_nodes)
#         # all_points = self.build_graph_batch(all_points, edge_index)
# # 
#         # print("all_points.x", all_points.shape, edge_index.shape)

#         # GAT layers
#         # x = F.relu(self.gat1(all_points[0], edge_index))
#         # x = F.relu(self.gat2(x, edge_index))
#         # x = self.fc(x)


#         # Extract outputs for non-manifold points (M points)
#         # batch_size = batch.max().item() + 1
#         batch_size= pos.shape[0]
#         num_points = pos_non_manifold.size(2)
      
#         # occupancy_logits = x[-num_points:].reshape(batch_size, self.out_channels, num_points)  # [Batch size,out_channels, M]
#         # occupancy_logits = x[num_points:].reshape(batch_size, self.out_channels, 3000)  # [Batch size,out_channels, M]
        
#         # occupancy_logits = x[:,num_points:].reshape(batch_size, self.out_channels, 3000)  # [Batch size,out_channels, M]

#         # return occupancy_logits
#         occupancy_logits = x.permute(0, 2, 1)  # Shape [Batch size, Out channels, Num nodes]

#         return occupancy_logits 

# class GATOccupancyPredictor(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_heads=4, out_channels=2, k=16):
#         """
#         Initialize the GAT-based occupancy predictor.
#         :param in_channels: Number of input features per node (should be 3 for 3D points)
#         :param hidden_channels: Number of hidden units for GATConv
#         :param num_heads: Number of attention heads in GATConv
#         :param out_channels: Number of output features per node
#         :param k: Number of nearest neighbors for graph construction
#         """
#         super().__init__()
#         self.k = k
        
#         # Input features are 3D coordinates
#         self.input_transform = nn.Linear(3, hidden_channels)
        
#         # GAT layers
#         self.gat1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=True)
#         self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        
#         self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
#         self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)
        
#         # Shortcut connection needs to match dimensions
#         self.shortcut = nn.Linear(hidden_channels, hidden_channels * num_heads)
        
#         # Final layer to get desired output channels
#         self.final = nn.Linear(hidden_channels * num_heads, out_channels)
#         self.out_channels = out_channels

#     @torch.no_grad()
#     def build_fast_graph(self, points, batch_size, num_points):
#         edge_index = knn_graph(
#             points, 
#             k=self.k, 
#             batch=torch.repeat_interleave(
#                 torch.arange(batch_size, device=points.device), 
#                 num_points
#             )
#         )
#         return edge_index

#     def forward(self, data, spatial_only=False, spectral_only=False):
#         if spatial_only:
#             return data
            
#         pos = data["pos"]
#         pos_non_manifold = data["pos_non_manifold"]
        
#         # Combine points and reshape
#         all_points = torch.cat([pos, pos_non_manifold], dim=2)  # [B, C, N]
#         batch_size, _, num_points = all_points.shape
#         all_points = all_points.transpose(1, 2).reshape(-1, 3)  # [B*N, 3]
        
#         # Initial feature transform
#         x = self.input_transform(all_points)  # [B*N, hidden_channels]
#         identity = self.shortcut(x)  # [B*N, hidden_channels * num_heads]
        
#         # Build graph
#         edge_index = self.build_fast_graph(
#             all_points, 
#             batch_size, 
#             num_points
#         )
        
#         # GAT layers
#         x = self.gat1(x, edge_index)  # [B*N, hidden_channels * num_heads]
#         x = self.bn1(x)
#         x = F.relu(x)
        
#         x = self.gat2(x, edge_index)  # [B*N, hidden_channels * num_heads]
#         x = self.bn2(x)
        
#         # Add residual
#         x = F.relu(x + identity)
        
#         # Final prediction
#         x = self.final(x)  # [B*N, out_channels]
        
#         # Reshape back to original dimensions
#         x = x.view(batch_size, -1, self.out_channels)  # [B, N, out_channels]
#         x = x.permute(0, 2, 1)  # [B, out_channels, N]
        
#         return x

class GATOccupancyPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=4, out_channels=2, k=16):
        """
        Initialize the enhanced GAT-based occupancy predictor.
        :param in_channels: Number of input features per node (should be 3 for 3D points)
        :param hidden_channels: Number of hidden units for GATConv
        :param num_heads: Number of attention heads in GATConv
        :param out_channels: Number of output features per node
        :param k: Number of nearest neighbors for graph construction
        """
        super().__init__()
        self.k = k
        
        # Input features transformation
        self.input_transform = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # Encoder GAT layers
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels * 2, heads=num_heads, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2 * num_heads)
        
        self.gat3 = GATConv(hidden_channels * 2 * num_heads, hidden_channels * 2, heads=num_heads, concat=True)
        self.bn3 = nn.BatchNorm1d(hidden_channels * 2 * num_heads)
        
        # Decoder layers with skip connections
        self.dec1 = nn.Sequential(
            nn.Linear(hidden_channels * 2 * num_heads, hidden_channels * 2 * num_heads),
            nn.BatchNorm1d(hidden_channels * 2 * num_heads),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2 * num_heads, hidden_channels * num_heads),
            nn.BatchNorm1d(hidden_channels * num_heads),
            nn.ReLU()
        )
        
        # Shortcut connections
        self.shortcut1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * num_heads),
            nn.BatchNorm1d(hidden_channels * num_heads)
        )
        
        self.shortcut2 = nn.Sequential(
            nn.Linear(hidden_channels * num_heads, hidden_channels * 2 * num_heads),
            nn.BatchNorm1d(hidden_channels * 2 * num_heads)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * num_heads * 2, hidden_channels * num_heads),
            nn.BatchNorm1d(hidden_channels * num_heads),
            nn.ReLU(),
            nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads),
            nn.BatchNorm1d(hidden_channels * num_heads),
            nn.ReLU()
        )
        
        # Output layers
        self.final = nn.Sequential(
            nn.Linear(hidden_channels * num_heads, hidden_channels * num_heads // 2),
            nn.BatchNorm1d(hidden_channels * num_heads // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * num_heads // 2, out_channels)
        )
        
        self.out_channels = out_channels
        self.dropout = nn.Dropout(0.1)

    @torch.no_grad()
    def build_fast_graph(self, points, batch_size, num_points):
        edge_index = knn_graph(
            points, 
            k=self.k, 
            batch=torch.repeat_interleave(
                torch.arange(batch_size, device=points.device), 
                num_points
            )
        )
        return edge_index

    def forward(self, data, spatial_only=False, spectral_only=False):
        if spatial_only:
            return data
            
        # Use pos points
        points = data["pos"]  # [B, C, N]
        batch_size, _, num_points = points.shape
        points = points.transpose(1, 2).reshape(-1, 3)  # [B*N, 3]
        
        # Initial feature transform
        x = self.input_transform(points)  # [B*N, hidden_channels]
        identity1 = self.shortcut1(x)
        
        # Build graph
        edge_index = self.build_fast_graph(
            points, 
            batch_size, 
            points.size(0) // batch_size
        )
        
        # Encoder path
        x1 = self.gat1(x, edge_index)  # [B*N, hidden_channels * num_heads]
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        x1 = x1 + identity1  # Skip connection 1
        
        identity2 = self.shortcut2(x1)
        
        x2 = self.gat2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        x3 = self.gat3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.dropout(x3)
        x3 = x3 + identity2  # Skip connection 2
        
        # Decoder path
        x = self.dec1(x3)
        
        # Feature fusion with first skip connection
        x = torch.cat([x, x1], dim=1)
        x = self.fusion(x)
        
        # Final prediction
        x = self.final(x)
        
        # Reshape back to original dimensions
        x = x.view(batch_size, -1, self.out_channels)  # [B, N, out_channels]
        x = x.permute(0, 2, 1)  # [B, out_channels, N]
        
        return x
    
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
