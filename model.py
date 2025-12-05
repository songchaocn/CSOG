import torch.nn as nn
import torch
from torch.nn import init
from torch_geometric.nn import GATConv, MLP
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Source(nn.Module):
    def __init__(self, mlp_model, mirna_sample_gat_model, meth_sample_gat_model,
                 mrna_sample_gat_model, fusion_model):
        super(Source, self).__init__()
        self.mlp_model = mlp_model
        self.mirna_sample_gat_model = mirna_sample_gat_model
        self.meth_sample_gat_model = meth_sample_gat_model
        self.mrna_sample_gat_model = mrna_sample_gat_model
        self.fusion_model = fusion_model

    def forward(self, mrna_sample_data, meth_sample_data, mirna_sample_data):
        mirna_sample_out = self.mirna_sample_gat_model(mirna_sample_data.x, mirna_sample_data.edge_index)
        meth_sample_out = self.meth_sample_gat_model(meth_sample_data.x, meth_sample_data.edge_index)
        mrna_sample_out = self.mrna_sample_gat_model(mrna_sample_data.x, mrna_sample_data.edge_index)

        sample_out, _ = self.fusion_model(mrna_sample_out, meth_sample_out, mirna_sample_out)

        output = self.mlp_model(sample_out)

        return output, mrna_sample_out, meth_sample_out, mirna_sample_out, sample_out


class Target(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_classes):
        super(Target, self).__init__()
        self.conv1 = GATConv(input_size, hidden_size, heads=4, concat=False)

        self.conv2 = GATConv(hidden_size, hidden_size, heads=4, concat=False)
        self.conv3 = GATConv(hidden_size, output_size, heads=4, concat=False)
        self.residual_transform1 = nn.Linear(input_size, hidden_size)
        self.residual_transform2 = nn.Linear(input_size, output_size)
        self.mlp = MLP(in_channels=128, hidden_channels=16,
                       out_channels=num_classes, num_layers=2, dropout=0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        output = self.mlp(x)
        return output, x


class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=4, concat=False)
        self.conv2 = GATConv(h_feats, h_feats, heads=4, concat=False)
        self.conv3 = GATConv(h_feats, out_feats, heads=4, concat=False)
        self.residual_transform1 = nn.Linear(in_feats, h_feats)
        self.residual_transform2 = nn.Linear(in_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        torch.nn.init.xavier_uniform_(self.residual_transform1.weight)
        torch.nn.init.zeros_(self.residual_transform1.bias)
        torch.nn.init.xavier_uniform_(self.residual_transform2.weight)
        torch.nn.init.zeros_(self.residual_transform2.bias)

    def forward(self, x, edge_index):
        x_res1 = self.residual_transform1(x)
        x_res2 = self.residual_transform2(x)
        x = self.conv1(x, edge_index)
        x = x + x_res1
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = x + x_res1
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        return x + x_res2


class NodeEmbeddingMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeEmbeddingMLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = nn.Linear(hidden_channels, out_channels)
        self.fc5 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=0.0)
        self._initialize_weights()

    def _initialize_weights(self):
        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.fc1.bias)
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.fc2.bias)
        init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.fc3.bias)
        init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.fc4.bias)
        init.kaiming_uniform_(self.fc5.weight, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.fc5.bias)

    def forward(self, node_embedding):
        x = node_embedding
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FusionModel, self).__init__()
        self.attention_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.attention_y = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.attention_z = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y, z):
        score_x = self.attention_x(x)
        score_y = self.attention_y(y)
        score_z = self.attention_z(z)

        scores = torch.cat([score_x, score_y, score_z], dim=1)
        fusion_weights = F.softmax(scores, dim=1)

        x_weighted = fusion_weights[:, 0].unsqueeze(-1) * x
        y_weighted = fusion_weights[:, 1].unsqueeze(-1) * y
        z_weighted = fusion_weights[:, 2].unsqueeze(-1) * z

        fusion_result = x_weighted + y_weighted + z_weighted
        return fusion_result, fusion_weights


def graph_propagation_loss(embeddings, edge_index, num_nodes):
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    embeddings_propagated = torch.matmul(adj_matrix, embeddings)
    loss = torch.norm(embeddings_propagated - embeddings, p='fro') ** 2
    return loss


def contrastive_loss_robust(embeddings, pos_pairs, neg_pairs, temperature=1):
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf values!")

    pos_sim = F.cosine_similarity(embeddings[pos_pairs[:, 0]],
                                  embeddings[pos_pairs[:, 1]], dim=1)
    neg_sim = F.cosine_similarity(embeddings[neg_pairs[:, 0]],
                                  embeddings[neg_pairs[:, 1]], dim=1)

    pos_logits = pos_sim / temperature
    neg_logits = neg_sim / temperature

    max_logit = torch.max(torch.cat([pos_logits, neg_logits]))
    pos_exp = torch.exp(pos_logits - max_logit)
    neg_exp = torch.exp(neg_logits - max_logit)

    loss = -torch.log(pos_exp / (pos_exp + neg_exp.sum())) - max_logit

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print(f"Loss instability detected. pos_sim range: [{pos_sim.min():.4f}, {pos_sim.max():.4f}]")
        print(f"neg_sim range: [{neg_sim.min():.4f}, {neg_sim.max():.4f}]")
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return loss.mean()


def final_loss(outputs1, outputs2, edge_index, num_nodes, pos_pairs, neg_pairs, lambda_1=2e-7, lambda_2=0.2):
    reg_loss = graph_propagation_loss(outputs1, edge_index, num_nodes)
    cont_loss = contrastive_loss_robust(outputs2, pos_pairs, neg_pairs)
    total_loss = lambda_1 * reg_loss + lambda_2 * cont_loss
    return total_loss, lambda_1 * reg_loss, lambda_2 * cont_loss


def sourceModels(sample_parameters):
    mirna_sample_gat_model = GAT(sample_parameters.mirna_input_size, sample_parameters.mirna_hidden_size,
                                 sample_parameters.sample_output_size).to(device)
    meth_sample_gat_model = GAT(
        sample_parameters.mrna_meth_input_size, sample_parameters.mrna_meth_hidden_size,
        sample_parameters.sample_output_size).to(device)
    mrna_sample_gat_model = GAT(sample_parameters.mrna_meth_input_size, sample_parameters.mrna_meth_hidden_size,
                                sample_parameters.sample_output_size).to(device)

    mlp_model = NodeEmbeddingMLP(sample_parameters.sample_output_size, sample_parameters.mlp_hidden_size,
                                 sample_parameters.num_classes).to(device)
    fusion_model = FusionModel(sample_parameters.sample_output_size, 16).to(device)
    source = Source(mlp_model, mirna_sample_gat_model, meth_sample_gat_model, mrna_sample_gat_model, fusion_model)
    return source
