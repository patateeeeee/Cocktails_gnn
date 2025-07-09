"""
Graph Neural Network pour apprendre les embeddings d'ingrédients et cocktails
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

class CocktailGNN(nn.Module):
    """
    GNN pour apprendre les embeddings cocktails-ingrédients
    Utilise GraphSAGE pour gérer les graphes bipartites
    """
    
    def __init__(self, 
                 num_nodes: int,
                 input_dim: int = 2,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super(CocktailGNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Couche d'entrée pour transformer les features initiales
        self.input_transform = nn.Linear(input_dim, embedding_dim)
        
        # Couches GraphSAGE
        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGE(embedding_dim, hidden_dim, num_layers=1, dropout=dropout))
        
        for _ in range(num_layers - 1):
            self.convs.append(GraphSAGE(hidden_dim, hidden_dim, num_layers=1, dropout=dropout))
        
        # Couche de sortie pour la classification/prédiction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass du GNN"""
        
        # Transformer les features d'entrée
        x = self.input_transform(x)
        x = F.relu(x)
        
        # Message passing avec GraphSAGE
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
    
    def predict_compatibility(self, node_embeddings, cocktail_idx, ingredient_idx):
        """Prédit si un cocktail et un ingrédient vont bien ensemble"""
        cocktail_emb = node_embeddings[cocktail_idx]
        ingredient_emb = node_embeddings[ingredient_idx]
        
        # Similarité cosinus ou produit scalaire
        similarity = torch.sum(cocktail_emb * ingredient_emb, dim=1)
        return torch.sigmoid(similarity)

class CocktailDataset:
    """Dataset pour entraîner le GNN sur les relations cocktail-ingrédient"""
    
    def __init__(self, graph_path: str = "data/processed/cocktail_graph.gml"):
        self.graph = nx.read_gml(graph_path)
        self.prepare_data()
    
    def prepare_data(self):
        """Prépare les données pour PyTorch Geometric"""
        
        # Mapping nœuds → indices
        self.nodes = list(self.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        
        # Séparer cocktails et ingrédients
        self.cocktail_nodes = [node for node in self.nodes 
                             if self.graph.nodes[node].get('node_type') == 'cocktail']
        self.ingredient_nodes = [node for node in self.nodes 
                               if self.graph.nodes[node].get('node_type') == 'ingredient']
        
        self.cocktail_indices = [self.node_to_idx[node] for node in self.cocktail_nodes]
        self.ingredient_indices = [self.node_to_idx[node] for node in self.ingredient_nodes]
        
        print(f"📊 Dataset préparé:")
        print(f"   - {len(self.cocktail_nodes)} cocktails")
        print(f"   - {len(self.ingredient_nodes)} ingrédients")
        print(f"   - {len(self.nodes)} nœuds total")
        
        # Créer features initiales (one-hot basique)
        self.node_features = self._create_node_features()
        
        # Créer edge_index pour PyTorch Geometric
        self.edge_index = self._create_edge_index()
        
        # Créer les échantillons d'entraînement (link prediction)
        self.train_samples, self.test_samples = self._create_training_samples()
    
    def _create_node_features(self) -> torch.Tensor:
        """Crée des features initiales pour les nœuds"""
        num_nodes = len(self.nodes)
        features = torch.zeros(num_nodes, 2)  # [is_cocktail, is_ingredient]
        
        for i, node in enumerate(self.nodes):
            if node in self.cocktail_nodes:
                features[i, 0] = 1.0  # cocktail
            else:
                features[i, 1] = 1.0  # ingrédient
                # Ajouter fréquence si disponible
                freq = self.graph.nodes[node].get('frequency', 1)
                features[i, 1] = min(freq / 10.0, 1.0)  # normaliser
        
        return features
    
    def _create_edge_index(self) -> torch.Tensor:
        """Convertit le graphe NetworkX en format PyTorch Geometric"""
        edges = []
        
        for edge in self.graph.edges():
            u, v = edge
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            
            # Ajouter les deux directions (graphe non-dirigé)
            edges.append([u_idx, v_idx])
            edges.append([v_idx, u_idx])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _create_training_samples(self, test_ratio: float = 0.2) -> Tuple[List, List]:
        """Crée des échantillons pour l'entraînement en link prediction"""
        
        # Échantillons positifs (liens existants)
        positive_samples = []
        for cocktail in self.cocktail_nodes:
            for ingredient in self.graph.neighbors(cocktail):
                if ingredient in self.ingredient_nodes:
                    cocktail_idx = self.node_to_idx[cocktail]
                    ingredient_idx = self.node_to_idx[ingredient]
                    weight = self.graph[cocktail][ingredient].get('weight', 1.0)
                    
                    positive_samples.append({
                        'cocktail_idx': cocktail_idx,
                        'ingredient_idx': ingredient_idx,
                        'label': 1,
                        'weight': weight
                    })
        
        # Échantillons négatifs (liens inexistants)
        negative_samples = []
        np.random.seed(42)
        
        for _ in range(len(positive_samples)):
            # Choisir cocktail et ingrédient aléatoires
            cocktail = np.random.choice(self.cocktail_nodes)
            ingredient = np.random.choice(self.ingredient_nodes)
            
            # Vérifier qu'il n'y a pas de lien
            if not self.graph.has_edge(cocktail, ingredient):
                cocktail_idx = self.node_to_idx[cocktail]
                ingredient_idx = self.node_to_idx[ingredient]
                
                negative_samples.append({
                    'cocktail_idx': cocktail_idx,
                    'ingredient_idx': ingredient_idx,
                    'label': 0,
                    'weight': 0.0
                })
        
        # Mélanger et séparer train/test
        all_samples = positive_samples + negative_samples
        np.random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * (1 - test_ratio))
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]
        
        print(f"📋 Échantillons créés:")
        print(f"   - {len(positive_samples)} positifs")
        print(f"   - {len(negative_samples)} négatifs") 
        print(f"   - {len(train_samples)} train, {len(test_samples)} test")
        
        return train_samples, test_samples
    
    def get_pytorch_geometric_data(self) -> Data:
        """Retourne les données au format PyTorch Geometric"""
        return Data(
            x=self.node_features,
            edge_index=self.edge_index,
            num_nodes=len(self.nodes)
        )

class GNNTrainer:
    """Entraîneur pour le GNN cocktail"""
    
    def __init__(self, dataset: CocktailDataset, device: str = 'cpu'):
        self.dataset = dataset
        self.device = device
        
        # Modèle
        self.model = CocktailGNN(
            num_nodes=len(dataset.nodes),
            input_dim=dataset.node_features.shape[1],  # Dimension des features d'entrée
            embedding_dim=128,
            hidden_dim=256,
            num_layers=3
        ).to(device)
        
        # Optimiseur
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)  # 100x plus petit
        self.criterion = nn.BCELoss()
        
        # Données PyTorch Geometric
        self.data = dataset.get_pytorch_geometric_data().to(device)
        
        # Historique d'entraînement
        self.history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    def train_epoch(self) -> float:
        """Entraîne le modèle pour une époque"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Mélanger les échantillons pour chaque époque
        import random
        train_samples = self.dataset.train_samples.copy()
        random.shuffle(train_samples)
        
        for sample in train_samples:
            self.optimizer.zero_grad()
            
            # Récupérer les embeddings pour chaque échantillon
            node_embeddings = self.model(self.data.x, self.data.edge_index)
            
            cocktail_idx = sample['cocktail_idx']
            ingredient_idx = sample['ingredient_idx']
            label = torch.tensor([float(sample['label'])], device=self.device)
            
            # Prédiction
            pred = self.model.predict_compatibility(
                node_embeddings, 
                torch.tensor([cocktail_idx], device=self.device),
                torch.tensor([ingredient_idx], device=self.device)
            )
            
            # Loss
            loss = self.criterion(pred, label)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = (pred > 0.5).float()
            correct += (predicted == label).sum().item()
            total += 1
        
        avg_loss = total_loss / len(train_samples)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self) -> float:
        """Évalue le modèle sur le test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            node_embeddings = self.model(self.data.x, self.data.edge_index)
            
            for sample in self.dataset.test_samples:
                cocktail_idx = sample['cocktail_idx']
                ingredient_idx = sample['ingredient_idx']
                label = sample['label']
                
                pred = self.model.predict_compatibility(
                    node_embeddings,
                    torch.tensor([cocktail_idx], device=self.device),
                    torch.tensor([ingredient_idx], device=self.device)
                )
                
                predicted = (pred > 0.5).float().cpu().item()
                correct += (predicted == label)
                total += 1
        
        return correct / total
    
    def train(self, epochs: int = 100):
        """Entraîne le modèle"""
        print(f"🚀 Début de l'entraînement ({epochs} époques)")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            
            if epoch % 10 == 0:
                test_acc = self.evaluate()
                print(f"Époque {epoch:3d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")
                
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['test_acc'].append(test_acc)
        
        final_test_acc = self.evaluate()
        print(f"\n✅ Entraînement terminé ! Accuracy finale: {final_test_acc:.3f}")
        
        return self.history
    
    def plot_training_history(self):
        """Affiche l'historique d'entraînement"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Époque (x10)')
        ax1.set_ylabel('Loss')
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['test_acc'], label='Test')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Époque (x10)')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path: str = "data/processed/cocktail_gnn_model.pth"):
        """Sauvegarde le modèle entraîné"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'node_to_idx': self.dataset.node_to_idx,
            'cocktail_nodes': self.dataset.cocktail_nodes,
            'ingredient_nodes': self.dataset.ingredient_nodes,
            'history': self.history
        }, path)
        print(f"💾 Modèle sauvegardé: {path}")

def main():
    """Test de l'entraînement du GNN"""
    print("🧠 Cocktail GNN - Entraînement du modèle")
    print("=" * 50)
    
    # Charger les données
    dataset = CocktailDataset()
    
    # Créer l'entraîneur
    trainer = GNNTrainer(dataset)
    
    # Entraîner
    history = trainer.train(epochs=300)
    
    # Visualiser les résultats
    trainer.plot_training_history()
    
    # Sauvegarder
    trainer.save_model()
    
    print("\n✅ Modèle GNN entraîné et sauvegardé !")

if __name__ == "__main__":
    main()