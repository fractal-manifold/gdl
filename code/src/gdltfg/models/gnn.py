import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class GraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(GraphConvolutionalNetwork, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=-1)


class GraphAttentionNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=8, dropout=0.5):
        super(GraphAttentionNetwork, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=-1)


class GNNTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                pred = self.model(data.x, data.edge_index, data.batch).argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
        return correct / total
    
    def train(self, train_loader, val_loader, epochs=100):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_acc = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            self.scheduler.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        ax2.plot(self.val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()


def load_graph_dataset(name='MUTAG', use_synthetic=False):
    if use_synthetic:
        from ..datasets.synthetic import SyntheticGraphDataset
        dataset = SyntheticGraphDataset(num_graphs=500, num_classes=3)
        num_node_features = dataset[0].x.shape[1]
        num_classes = 3
    else:
        dataset = TUDataset(root='/tmp/' + name, name=name)
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
    
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, num_node_features, num_classes


def train_gcn_model(use_synthetic=True):
    print("Entrenando Graph Convolutional Network (GCN)")
    train_loader, val_loader, num_features, num_classes = load_graph_dataset('MUTAG', use_synthetic=use_synthetic)
    
    model = GraphConvolutionalNetwork(
        input_dim=num_features,
        hidden_dim=64,
        output_dim=num_classes
    )
    
    trainer = GNNTrainer(model)
    trainer.train(train_loader, val_loader, epochs=50)
    trainer.plot_training()
    
    return model, trainer


def train_gat_model(use_synthetic=True):
    print("Entrenando Graph Attention Network (GAT)")
    train_loader, val_loader, num_features, num_classes = load_graph_dataset('MUTAG', use_synthetic=use_synthetic)
    
    model = GraphAttentionNetwork(
        input_dim=num_features,
        hidden_dim=32,
        output_dim=num_classes,
        heads=4
    )
    
    trainer = GNNTrainer(model)
    trainer.train(train_loader, val_loader, epochs=50)
    trainer.plot_training()
    
    return model, trainer


if __name__ == "__main__":
    print("=== Modelos de Redes de Grafos (GNN) ===")
    
    gcn_model, gcn_trainer = train_gcn_model()
    print(f"GCN - Precisión final: {gcn_trainer.val_accuracies[-1]:.4f}")
    
    gat_model, gat_trainer = train_gat_model()
    print(f"GAT - Precisión final: {gat_trainer.val_accuracies[-1]:.4f}")