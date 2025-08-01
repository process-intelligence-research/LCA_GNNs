import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

# Model structure
from torch.nn import ELU, GRU, Dropout, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import NNConv
from torch_scatter import scatter_add, scatter_max, scatter_mean


class qspr(torch.nn.Module):
    def __init__(
        self,
        out_feature: int,
        input_features: int = 56,
        init_weights: bool = True,
        dropout: int = 0.1,
    ):
        """
        Neural network model for QSPR (Quantitative Structure-Property Relationship) prediction.

        This model uses a simple multi-layer perceptron (MLP) to process molecular features
        and predict properties based on the input molecular descriptors.

        Args:
            out_feature (int): Number of output features/targets to predict
            input_features (int, optional): Number of input molecular features. Defaults to 56.
            init_weights (bool, optional): If True, weights are manually initialized using
                normal distribution (mean=0, std=0.1) for Linear layers and zero bias.
                Defaults to True.
            dropout (float, optional): Dropout rate for regularization. Currently not used
                in the model architecture. Defaults to 0.1.
        """
        super(qspr, self).__init__()

        self.input_features = input_features
        self.output = out_feature
        self.mlp = Sequential(
            Linear(self.input_features, 16),
            ReLU(),
            Linear(16, 16),
            ReLU(),
            Linear(16, self.output),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, data: torch_geometric.data.batch.Data):
        """
        Forward pass through the neural network.

        Processes the input molecular features through the MLP to generate predictions.

        Args:
            data (torch_geometric.data.batch.Data): Batch of data containing molecular features.
                Uses data.x which should be a tensor of shape (batch_size, input_features)
                containing the molecular descriptors.

        Returns
        -------
            torch.Tensor: Predictions with shape (batch_size, out_feature) containing
            the predicted property values for each molecule in the batch.
        """
        # Process the node features
        x = self.mlp(data.x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)


class GNN_M(torch.nn.Module):
    def __init__(
        self,
        out_feature,
        num_features=32,
        dim=64,
        edge_dim=13,
        inte_dim=128,
        country_dim=91,
        conv_type=3,
        pool_type="add",
        init_weights: bool = True,
    ):
        """
        Graph Neural Network model for molecular property prediction.

        This model uses Neural Network Convolution (NNConv) layers combined with GRU cells
        to process molecular graphs. It transforms node and edge features through multiple
        convolution layers, applies graph-level pooling, and uses fully connected layers
        for final prediction.

        Architecture:
        1. Linear transformation of node features
        2. Multiple NNConv + GRU layers for message passing
        3. Graph-level pooling (add/mean/max)
        4. Fully connected layers for final prediction

        Args:
            out_feature (int): Number of output features/targets to predict
            num_features (int, optional): Number of input node features. Defaults to 32.
            dim (int, optional): Hidden dimension for node embeddings. Defaults to 64.
            edge_dim (int, optional): Number of edge features. Defaults to 13.
            inte_dim (int, optional): Intermediate dimension for edge network. Defaults to 128.
            country_dim (int, optional): Country dimension (currently not used in forward pass).
                Defaults to 91.
            conv_type (int, optional): Number of convolution layers. Defaults to 3.
            pool_type (str, optional): Type of graph-level pooling. Options: 'add', 'mean', 'max'.
                Defaults to 'add'.
            init_weights (bool, optional): If True, weights are manually initialized using
                normal distribution (mean=0, std=0.1) for Linear layers and zero bias.
                Defaults to True.
        """
        super(GNN_M, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.edge_dim = edge_dim
        self.inte_dim = inte_dim
        self.country_dim = country_dim
        self.conv_type = conv_type
        self.pool_type = pool_type

        self.lin0 = torch.nn.Linear(self.num_features, self.dim)

        self.nn = Sequential(
            Linear(self.edge_dim, self.inte_dim),
            ReLU(),
            Linear(self.inte_dim, self.dim * self.dim),
        )
        self.conv = NNConv(self.dim, self.dim, self.nn, aggr="add")
        self.gru = GRU(self.dim, self.dim)

        self.fc1 = torch.nn.Linear(self.dim, 256)
        self.fc11 = torch.nn.Linear(256, 128)
        self.fc12 = torch.nn.Linear(128, 64)
        self.fc13 = torch.nn.Linear(64, out_feature)
        if init_weights:
            self._initialize_weights()

    def forward(self, data):
        """
        Forward pass through the Graph Neural Network.

        Processes molecular graphs through the following steps:
        1. Transform node features using linear layer
        2. Apply multiple NNConv + GRU layers for iterative message passing
        3. Pool node representations to graph-level representation
        4. Apply fully connected layers for final prediction

        Parameters
        ----------
        data : torch_geometric.data.Data or torch_geometric.data.Batch
            Graph data containing:
            - data.x: Node feature matrix of shape (num_nodes, num_features)
            - data.edge_index: Edge connectivity matrix of shape (2, num_edges)
            - data.edge_attr: Edge feature matrix of shape (num_edges, edge_dim)
            - data.batch: Batch vector assigning each node to a graph (for batched processing)

        Returns
        -------
        torch.Tensor
            Predicted molecular properties of shape (batch_size, out_feature)
        """
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        ## process the edge features and combine with the node featuresbb
        for i in range(self.conv_type):
            m = F.elu(self.conv(out, data.edge_index.long(), data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        x = scatter_add(out, data.batch, dim=0)

        if self.pool_type == "mean":
            x = scatter_mean(out, data.batch, dim=0)
        if self.pool_type == "max":
            x = scatter_max(out, data.batch, dim=0)[0]
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc11(x))
        x = F.elu(self.fc12(x))
        # x = F.elu(self.fc13(x))
        x = self.fc13(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)


class GNN_C_single(torch.nn.Module):
    def __init__(
        self,
        num_features=32,
        dim=64,
        edge_dim=13,
        inte_dim=128,
        country_dim=91,
        conv_type=3,
        pool_type="add",
        init_weights: bool = True,
    ):
        """
        Graph Neural Network model for molecular property prediction with country information.

        This model extends the basic GNN architecture by incorporating country-specific features
        in the final prediction layer. It uses Neural Network Convolution (NNConv) layers
        combined with GRU cells to process molecular graphs, then concatenates country information
        before the final fully connected layers.

        Architecture:
        1. Linear transformation of node features
        2. Multiple NNConv + GRU layers for message passing
        3. Graph-level pooling (add/mean/max)
        4. Concatenation with country features
        5. Fully connected layers for final prediction (outputs single value)

        Parameters
        ----------
        num_features : int, optional
            Number of input node features. Defaults to 32.
        dim : int, optional
            Hidden dimension for node embeddings. Defaults to 64.
        edge_dim : int, optional
            Number of edge features. Defaults to 13.
        inte_dim : int, optional
            Intermediate dimension for edge network. Defaults to 128.
        country_dim : int, optional
            Dimension of country feature vector. Defaults to 91.
        conv_type : int, optional
            Number of convolution layers. Defaults to 3.
        pool_type : str, optional
            Type of graph-level pooling. Options: 'add', 'mean', 'max'.
            Defaults to 'add'.
        init_weights : bool, optional
            If True, weights are manually initialized using normal distribution
            (mean=0, std=0.1) for Linear layers and zero bias. Defaults to True.
        """
        super(GNN_C_single, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.edge_dim = edge_dim
        self.inte_dim = inte_dim
        self.country_dim = country_dim
        self.conv_type = conv_type
        self.pool_type = pool_type

        self.lin0 = torch.nn.Linear(self.num_features, self.dim)

        self.nn = Sequential(
            Linear(self.edge_dim, self.inte_dim),
            ReLU(),
            Linear(self.inte_dim, self.dim * self.dim),
        )
        self.conv = NNConv(self.dim, self.dim, self.nn, aggr="add")
        self.gru = GRU(self.dim, self.dim)
        self.fc1 = torch.nn.Linear(self.dim + self.country_dim, 256)
        self.fc11 = torch.nn.Linear(256, 128)
        self.fc12 = torch.nn.Linear(128, 64)
        self.fc13 = torch.nn.Linear(64, 1)

        if init_weights:
            self._initialize_weights()

    def forward(self, data):
        """
        Forward pass through the Graph Neural Network with country features.

        Processes molecular graphs and incorporates country-specific information:
        1. Transform node features using linear layer
        2. Apply multiple NNConv + GRU layers for iterative message passing
        3. Pool node representations to graph-level representation
        4. Concatenate with country features
        5. Apply fully connected layers for final single-value prediction

        Parameters
        ----------
        data : torch_geometric.data.Data or torch_geometric.data.Batch
            Graph data containing:
            - data.x: Node feature matrix of shape (num_nodes, num_features)
            - data.edge_index: Edge connectivity matrix of shape (2, num_edges)
            - data.edge_attr: Edge feature matrix of shape (num_edges, edge_dim)
            - data.batch: Batch vector assigning each node to a graph (for batched processing)
            - data.country_id: Country feature vector of shape (batch_size, country_dim)

        Returns
        -------
        torch.Tensor
            Single predicted value per graph of shape (batch_size, 1)
        """
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        ## process the edge features and combine with the node featuresbb
        for i in range(self.conv_type):
            m = F.elu(self.conv(out, data.edge_index.long(), data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        x = scatter_add(out, data.batch, dim=0)

        if self.pool_type == "mean":
            x = scatter_mean(out, data.batch, dim=0)
        if self.pool_type == "max":
            x = scatter_max(out, data.batch, dim=0)[0]
        x = torch.cat((x, data.country_id), 1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc11(x))
        x = F.elu(self.fc12(x))
        x = self.fc13(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)


class GNN_E_single(torch.nn.Module):
    def __init__(
        self,
        num_features=32,
        dim=64,
        edge_dim=13,
        inte_dim=128,
        energy_dim=7,
        conv_type=3,
        pool_type="add",
        init_weights: bool = True,
    ):
        """
        Graph Neural Network model for molecular property prediction with energy information.

        This model extends the basic GNN architecture by incorporating energy-specific features
        in the final prediction layer. It uses Neural Network Convolution (NNConv) layers
        combined with GRU cells to process molecular graphs, then concatenates energy information
        and applies a regularized MLP with dropout for final prediction.

        Architecture:
        1. Linear transformation of node features
        2. Multiple NNConv + GRU layers for message passing
        3. Graph-level pooling (add/mean/max)
        4. Concatenation with energy features
        5. Regularized MLP with ELU activation and dropout for final prediction

        Parameters
        ----------
        num_features : int, optional
            Number of input node features. Defaults to 32.
        dim : int, optional
            Hidden dimension for node embeddings. Defaults to 64.
        edge_dim : int, optional
            Number of edge features. Defaults to 13.
        inte_dim : int, optional
            Intermediate dimension for edge network. Defaults to 128.
        energy_dim : int, optional
            Dimension of energy feature vector. Defaults to 7.
        conv_type : int, optional
            Number of convolution layers. Defaults to 3.
        pool_type : str, optional
            Type of graph-level pooling. Options: 'add', 'mean', 'max'.
            Defaults to 'add'.
        init_weights : bool, optional
            If True, weights are manually initialized using normal distribution
            (mean=0, std=0.1) for Linear layers and zero bias. Defaults to True.
        """
        super(GNN_E_single, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.edge_dim = edge_dim
        self.inte_dim = inte_dim
        self.energy_dim = energy_dim
        self.conv_type = conv_type
        self.pool_type = pool_type

        self.lin0 = torch.nn.Linear(self.num_features, self.dim)

        self.nn = Sequential(
            Linear(self.edge_dim, self.inte_dim),
            ReLU(),
            Linear(self.inte_dim, self.dim * self.dim),
        )
        self.conv = NNConv(self.dim, self.dim, self.nn, aggr="add")
        self.gru = GRU(self.dim, self.dim)
        self.mlp = Sequential(
            Linear(self.dim + self.energy_dim, 256),
            ELU(),
            Dropout(),
            Linear(256, 128),
            ELU(),
            Dropout(),
            Linear(128, 64),
            ELU(),
            Dropout(),
            Linear(64, 1),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, data):
        """
        Forward pass through the Graph Neural Network with energy features.

        Processes molecular graphs and incorporates energy-specific information:
        1. Transform node features using linear layer
        2. Apply multiple NNConv + GRU layers for iterative message passing
        3. Pool node representations to graph-level representation
        4. Concatenate with energy features
        5. Apply regularized MLP with dropout for final single-value prediction

        Parameters
        ----------
        data : torch_geometric.data.Data or torch_geometric.data.Batch
            Graph data containing:
            - data.x: Node feature matrix of shape (num_nodes, num_features)
            - data.edge_index: Edge connectivity matrix of shape (2, num_edges)
            - data.edge_attr: Edge feature matrix of shape (num_edges, edge_dim)
            - data.batch: Batch vector assigning each node to a graph (for batched processing)
            - data.energy_id: Energy feature vector of shape (batch_size, energy_dim)

        Returns
        -------
        torch.Tensor
            Single predicted value per graph of shape (batch_size, 1)
        """
        ### process the node features
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        ## process the edge features and combine with the node featuresbb
        for i in range(self.conv_type):
            m = F.elu(self.conv(out, data.edge_index.long(), data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        x = scatter_add(out, data.batch, dim=0)

        if self.pool_type == "mean":
            x = scatter_mean(out, data.batch, dim=0)
        if self.pool_type == "max":
            x = scatter_max(out, data.batch, dim=0)[0]
        x = torch.cat((x, data.energy_id), 1)
        x = self.mlp(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)


class GNN_C_multi(torch.nn.Module):
    def __init__(
        self,
        num_features=32,
        dim=64,
        edge_dim=13,
        inte_dim=128,
        country_dim=91,
        conv_type=3,
        pool_type="add",
        sub_impact: tuple = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        init_weights: bool = True,
    ):
        """
        Multi-output Graph Neural Network for molecular property prediction with country information.

        This model extends the GNN architecture to predict multiple target values simultaneously
        using parallel MLPs. Each MLP in the ensemble predicts a different number of outputs
        as specified by the sub_impact tuple. The model incorporates country-specific features
        and uses regularized MLPs with dropout for robust multi-target prediction.

        Architecture:
        1. Linear transformation of node features
        2. Multiple NNConv + GRU layers for message passing
        3. Graph-level pooling (add/mean/max)
        4. Concatenation with country features
        5. Parallel MLPs with different output dimensions
        6. Horizontal concatenation of all MLP outputs

        Parameters
        ----------
        num_features : int, optional
            Number of input node features. Defaults to 32.
        dim : int, optional
            Hidden dimension for node embeddings. Defaults to 64.
        edge_dim : int, optional
            Number of edge features. Defaults to 13.
        inte_dim : int, optional
            Intermediate dimension for edge network. Defaults to 128.
        country_dim : int, optional
            Dimension of country feature vector. Defaults to 91.
        conv_type : int, optional
            Number of convolution layers. Defaults to 3.
        pool_type : str, optional
            Type of graph-level pooling. Options: 'add', 'mean', 'max'.
            Defaults to 'add'.
        sub_impact : tuple, optional
            Tuple specifying the number of outputs for each parallel MLP.
            Length determines number of parallel MLPs.
            Defaults to (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1).
        init_weights : bool, optional
            If True, weights are manually initialized using normal distribution
            (mean=0, std=0.1) for Linear layers and zero bias. Defaults to True.
        """
        super(GNN_C_multi, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.edge_dim = edge_dim
        self.inte_dim = inte_dim
        self.country_dim = country_dim
        self.conv_type = conv_type
        self.pool_type = pool_type

        self.lin0 = torch.nn.Linear(self.num_features, self.dim)

        self.nn = Sequential(
            Linear(self.edge_dim, self.inte_dim),
            ReLU(),
            Linear(self.inte_dim, self.dim * self.dim),
        )
        self.conv = NNConv(self.dim, self.dim, self.nn, aggr="add")
        self.gru = GRU(self.dim, self.dim)
        self.mlp = ModuleList([])  # generate list of parallel mlps based on setup
        for i in sub_impact:
            self.mlp.append(
                Sequential(
                    Linear(self.dim + self.country_dim, 256),
                    ELU(),
                    Dropout(),
                    Linear(256, 128),
                    ELU(),
                    Dropout(),
                    Linear(128, 64),
                    ELU(),
                    Dropout(),
                    Linear(64, i),
                )
            )

        if init_weights:
            self._initialize_weights()

    def forward(self, data):
        """
        Forward pass through the multi-output Graph Neural Network with country features.

        Processes molecular graphs and generates multiple predictions using parallel MLPs:
        1. Transform node features using linear layer
        2. Apply multiple NNConv + GRU layers for iterative message passing
        3. Pool node representations to graph-level representation
        4. Concatenate with country features
        5. Apply each parallel MLP to the same input
        6. Horizontally concatenate all MLP outputs

        Parameters
        ----------
        data : torch_geometric.data.Data or torch_geometric.data.Batch
            Graph data containing:
            - data.x: Node feature matrix of shape (num_nodes, num_features)
            - data.edge_index: Edge connectivity matrix of shape (2, num_edges)
            - data.edge_attr: Edge feature matrix of shape (num_edges, edge_dim)
            - data.batch: Batch vector assigning each node to a graph (for batched processing)
            - data.country_id: Country feature vector of shape (batch_size, country_dim)

        Returns
        -------
        torch.Tensor
            Concatenated predictions from all parallel MLPs with shape
            (batch_size, sum(sub_impact)). Each MLP contributes sub_impact[i] outputs.
        """
        ### process the node features
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        ## process the edge features and combine with the node featuresbb
        for i in range(self.conv_type):
            m = F.elu(self.conv(out, data.edge_index.long(), data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        x = scatter_add(out, data.batch, dim=0)

        if self.pool_type == "mean":
            x = scatter_mean(out, data.batch, dim=0)
        if self.pool_type == "max":
            x = scatter_max(out, data.batch, dim=0)[0]
        x = torch.cat((x, data.country_id), 1)
        ## add country label
        x_parallel = self.mlp[0](x)
        for i in self.mlp[1:]:
            x_parallel = torch.hstack((x_parallel, i(x)))

        return x_parallel

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)


class GNN_E_multi(torch.nn.Module):
    def __init__(
        self,
        num_features=32,
        dim=64,
        edge_dim=13,
        inte_dim=128,
        energy_dim=7,
        conv_type=3,
        pool_type="add",
        sub_impact: tuple = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        init_weights: bool = True,
    ):
        """
        Multi-output Graph Neural Network for molecular property prediction with energy information.

        This model combines the multi-target prediction capability of parallel MLPs with
        energy-specific features. It uses Neural Network Convolution (NNConv) layers combined
        with GRU cells to process molecular graphs, then concatenates energy information and
        applies multiple parallel regularized MLPs for simultaneous multi-target prediction.

        Architecture:
        1. Linear transformation of node features
        2. Multiple NNConv + GRU layers for message passing
        3. Graph-level pooling (add/mean/max)
        4. Concatenation with energy features
        5. Parallel regularized MLPs with different output dimensions
        6. Horizontal concatenation of all MLP outputs

        Parameters
        ----------
        num_features : int, optional
            Number of input node features. Defaults to 32.
        dim : int, optional
            Hidden dimension for node embeddings. Defaults to 64.
        edge_dim : int, optional
            Number of edge features. Defaults to 13.
        inte_dim : int, optional
            Intermediate dimension for edge network. Defaults to 128.
        energy_dim : int, optional
            Dimension of energy feature vector. Defaults to 7.
        conv_type : int, optional
            Number of convolution layers. Defaults to 3.
        pool_type : str, optional
            Type of graph-level pooling. Options: 'add', 'mean', 'max'.
            Defaults to 'add'.
        sub_impact : tuple, optional
            Tuple specifying the number of outputs for each parallel MLP.
            Length determines number of parallel MLPs.
            Defaults to (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1).
        init_weights : bool, optional
            If True, weights are manually initialized using normal distribution
            (mean=0, std=0.1) for Linear layers and zero bias. Defaults to True.
        """
        super(GNN_E_multi, self).__init__()
        self.num_features = num_features
        self.dim = dim
        self.edge_dim = edge_dim
        self.inte_dim = inte_dim
        self.energy_dim = energy_dim
        self.conv_type = conv_type
        self.pool_type = pool_type

        self.lin0 = torch.nn.Linear(self.num_features, self.dim)

        self.nn = Sequential(
            Linear(self.edge_dim, self.inte_dim),
            ReLU(),
            Linear(self.inte_dim, self.dim * self.dim),
        )
        self.conv = NNConv(self.dim, self.dim, self.nn, aggr="add")
        self.gru = GRU(self.dim, self.dim)
        self.mlp = ModuleList([])  # generate list of parallel mlps based on setup
        for i in sub_impact:
            self.mlp.append(
                Sequential(
                    Linear(self.dim + self.energy_dim, 256),
                    ELU(),
                    Dropout(),
                    Linear(256, 128),
                    ELU(),
                    Dropout(),
                    Linear(128, 64),
                    ELU(),
                    Dropout(),
                    Linear(64, i),
                )
            )

        if init_weights:
            self._initialize_weights()

    def forward(self, data):
        """
        Forward pass through the multi-output Graph Neural Network with energy features.

        Processes molecular graphs and generates multiple predictions using parallel MLPs
        with energy information:
        1. Transform node features using linear layer
        2. Apply multiple NNConv + GRU layers for iterative message passing
        3. Pool node representations to graph-level representation
        4. Concatenate with energy features
        5. Apply each parallel MLP to the same input
        6. Horizontally concatenate all MLP outputs

        Parameters
        ----------
        data : torch_geometric.data.Data or torch_geometric.data.Batch
            Graph data containing:
            - data.x: Node feature matrix of shape (num_nodes, num_features)
            - data.edge_index: Edge connectivity matrix of shape (2, num_edges)
            - data.edge_attr: Edge feature matrix of shape (num_edges, edge_dim)
            - data.batch: Batch vector assigning each node to a graph (for batched processing)
            - data.energy_id: Energy feature vector of shape (batch_size, energy_dim)

        Returns
        -------
        torch.Tensor
            Concatenated predictions from all parallel MLPs with shape
            (batch_size, sum(sub_impact)). Each MLP contributes sub_impact[i] outputs.
        """
        ### process the node features
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        ## process the edge features and combine with the node featuresbb
        for i in range(self.conv_type):
            m = F.elu(self.conv(out, data.edge_index.long(), data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        x = scatter_add(out, data.batch, dim=0)

        if self.pool_type == "mean":
            x = scatter_mean(out, data.batch, dim=0)
        if self.pool_type == "max":
            x = scatter_max(out, data.batch, dim=0)[0]
        x = torch.cat((x, data.energy_id), 1)
        ## add country label
        x_parallel = self.mlp[0](x)
        for i in self.mlp[1:]:
            x_parallel = torch.hstack((x_parallel, i(x)))

        return x_parallel

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)
