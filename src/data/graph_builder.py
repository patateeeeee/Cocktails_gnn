"""
Construction du graphe de cocktails et ingrédients
"""
import pandas as pd
import networkx as nx
import numpy as np
import ast
import re
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

class CocktailGraphBuilder:
    """Construit un graphe bipartite cocktails-ingrédients"""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.graph = nx.Graph()
        self.ingredients_vocab = {}
        self.cocktail_nodes = set()
        self.ingredient_nodes = set()
        
    def clean_ingredients(self) -> pd.DataFrame:
        """Nettoie et normalise les ingrédients"""
        print("🧹 Nettoyage des ingrédients...")
        
        # Convertir les strings en listes
        self.df['ingredients_list'] = self.df['ingredients'].apply(ast.literal_eval)
        self.df['measurements_list'] = self.df['measurements'].apply(ast.literal_eval)
        
        # Normaliser les noms d'ingrédients
        normalized_ingredients = []
        for ingredients in self.df['ingredients_list']:
            normalized = [self._normalize_ingredient(ing) for ing in ingredients]
            normalized_ingredients.append(normalized)
        
        self.df['ingredients_normalized'] = normalized_ingredients
        
        # Créer le vocabulaire d'ingrédients
        all_ingredients = []
        for ingredients in self.df['ingredients_normalized']:
            all_ingredients.extend(ingredients)
        
        ingredient_counts = Counter(all_ingredients)
        print(f"📊 Trouvé {len(ingredient_counts)} ingrédients uniques")
        
        # Garder seulement les ingrédients qui apparaissent plus d'une fois
        self.ingredients_vocab = {ing: count for ing, count in ingredient_counts.items() if count > 1}
        print(f"📊 Gardé {len(self.ingredients_vocab)} ingrédients fréquents")
        
        return self.df
    
    def _normalize_ingredient(self, ingredient: str) -> str:
        """Normalise un nom d'ingrédient"""
        # Minuscules et suppression espaces
        ingredient = ingredient.lower().strip()
        
        # Normalisation de synonymes courants
        synonyms = {
            'triple sec': 'cointreau',
            'lime juice': 'lime',
            'lemon juice': 'lemon', 
            'simple syrup': 'sugar syrup',
            'sweet vermouth': 'vermouth sweet',
            'dry vermouth': 'vermouth dry',
            'angostura bitters': 'bitters',
            'grenadine syrup': 'grenadine'
        }
        
        return synonyms.get(ingredient, ingredient)
    
    def parse_measurement(self, measurement: str) -> float:
        """Parse une mesure en valeur numérique standardisée (en ml)"""
        if not measurement or measurement == '':
            return 30.0  # valeur par défaut
            
        measurement = measurement.lower().strip()
        
        # Conversion des unités courantes en ml
        if 'oz' in measurement:
            # 1 oz = 29.5735 ml
            number = re.findall(r'[\d./]+', measurement)
            if number:
                return self._parse_fraction(number[0]) * 29.5735
        
        elif 'cl' in measurement:
            # 1 cl = 10 ml
            number = re.findall(r'[\d./]+', measurement)
            if number:
                return self._parse_fraction(number[0]) * 10
        
        elif 'ml' in measurement:
            number = re.findall(r'[\d./]+', measurement)
            if number:
                return self._parse_fraction(number[0])
        
        elif 'shot' in measurement:
            # 1 shot = ~44ml
            number = re.findall(r'[\d./]+', measurement)
            if number:
                return self._parse_fraction(number[0]) * 44
        
        elif 'dash' in measurement:
            return 2.0  # 1 dash ≈ 2ml
        
        elif 'part' in measurement:
            # Pour les proportions relatives
            number = re.findall(r'[\d./]+', measurement)
            if number:
                return self._parse_fraction(number[0]) * 30  # base 30ml par part
        
        # Si rien ne match, essayer d'extraire juste le nombre
        number = re.findall(r'[\d./]+', measurement)
        if number:
            return self._parse_fraction(number[0]) * 30  # assume 30ml par unité
        
        return 30.0  # défaut
    
    def _parse_fraction(self, fraction_str: str) -> float:
        """Parse une fraction comme '1/2' ou '1 1/2'"""
        if '/' not in fraction_str:
            return float(fraction_str)
        
        # Gestion des fractions mixtes comme "1 1/2"
        if ' ' in fraction_str:
            parts = fraction_str.split(' ')
            whole = float(parts[0])
            frac_parts = parts[1].split('/')
            frac = float(frac_parts[0]) / float(frac_parts[1])
            return whole + frac
        
        # Fraction simple
        parts = fraction_str.split('/')
        return float(parts[0]) / float(parts[1])
    
    def build_graph(self):
        """Construit le graphe bipartite cocktails-ingrédients"""
        print("🔗 Construction du graphe...")
        
        for idx, row in self.df.iterrows():
            cocktail_name = row['name']
            ingredients = row['ingredients_normalized']
            measurements = row['measurements_list']
            
            # Ajouter le nœud cocktail
            self.graph.add_node(cocktail_name, node_type='cocktail')
            self.cocktail_nodes.add(cocktail_name)
            
            # Ajouter les ingrédients et arêtes
            for ingredient, measurement in zip(ingredients, measurements):
                if ingredient in self.ingredients_vocab:
                    
                    # Ajouter le nœud ingrédient
                    if ingredient not in self.ingredient_nodes:
                        self.graph.add_node(ingredient, 
                                          node_type='ingredient',
                                          frequency=self.ingredients_vocab[ingredient])
                        self.ingredient_nodes.add(ingredient)
                    
                    # Parser la mesure
                    quantity_ml = self.parse_measurement(measurement)
                    
                    # Ajouter l'arête cocktail-ingrédient
                    self.graph.add_edge(cocktail_name, ingredient, 
                                      weight=quantity_ml,
                                      measurement_raw=measurement)
        
        print(f"📊 Graphe construit:")
        print(f"   - {len(self.cocktail_nodes)} cocktails")
        print(f"   - {len(self.ingredient_nodes)} ingrédients") 
        print(f"   - {self.graph.number_of_edges()} relations")
    
    def get_graph_stats(self) -> Dict:
        """Statistiques du graphe"""
        stats = {
            'nb_cocktails': len(self.cocktail_nodes),
            'nb_ingredients': len(self.ingredient_nodes),
            'nb_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_ingredients_per_cocktail': self.graph.number_of_edges() / len(self.cocktail_nodes)
        }
        
        # Top ingrédients par connectivité
        ingredient_degrees = [(node, self.graph.degree(node)) 
                            for node in self.ingredient_nodes]
        ingredient_degrees.sort(key=lambda x: x[1], reverse=True)
        stats['top_ingredients'] = ingredient_degrees[:10]
        
        return stats
    
    def visualize_graph_sample(self, max_nodes: int = 30):
        """Visualise un échantillon du graphe"""
        # Prendre un sous-ensemble pour la visualisation
        sample_cocktails = list(self.cocktail_nodes)[:10]
        sample_ingredients = []
        
        for cocktail in sample_cocktails:
            ingredients = list(self.graph.neighbors(cocktail))
            sample_ingredients.extend(ingredients)
        
        sample_ingredients = list(set(sample_ingredients))[:20]
        sample_nodes = sample_cocktails + sample_ingredients
        
        subgraph = self.graph.subgraph(sample_nodes)
        
        plt.figure(figsize=(15, 10))
        
        # Positions des nœuds
        pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Couleurs par type
        node_colors = ['lightblue' if node in self.cocktail_nodes else 'lightcoral' 
                      for node in subgraph.nodes()]
        
        # Dessiner le graphe
        nx.draw(subgraph, pos, 
                node_color=node_colors,
                node_size=500,
                font_size=8,
                font_weight='bold',
                with_labels=True,
                edge_color='gray',
                alpha=0.7)
        
        plt.title("Échantillon du graphe Cocktails-Ingrédients")
        
        # Créer une légende manuelle pour éviter les conflits
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='lightblue', label='Cocktails')
        red_patch = mpatches.Patch(color='lightcoral', label='Ingrédients')
        plt.legend(handles=[blue_patch, red_patch], loc='upper right')
        
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()
    
    def save_graph(self, filepath: str = "data/processed/cocktail_graph.gml"):
        """Sauvegarde le graphe"""
        nx.write_gml(self.graph, filepath)
        print(f"💾 Graphe sauvegardé: {filepath}")
    
    def save_adjacency_matrix(self, filepath: str = "data/processed/adjacency_matrix.csv"):
        """Sauvegarde la matrice d'adjacence pour les GNN"""
        # Créer mapping des nœuds vers indices
        all_nodes = list(self.graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Matrice d'adjacence
        adj_matrix = nx.adjacency_matrix(self.graph, nodelist=all_nodes)
        
        # Sauvegarde avec métadonnées (structure correcte pour JSON)
        metadata = {
            'total_nodes': len(all_nodes),
            'nb_cocktails': len(self.cocktail_nodes),
            'nb_ingredients': len(self.ingredient_nodes),
            'cocktail_indices': [i for i, node in enumerate(all_nodes) if node in self.cocktail_nodes],
            'ingredient_indices': [i for i, node in enumerate(all_nodes) if node in self.ingredient_nodes]
        }
        
        # Sauvegarde séparée des noms de nœuds et types
        node_info = []
        for i, node in enumerate(all_nodes):
            node_info.append({
                'index': i,
                'name': node,
                'type': self.graph.nodes[node].get('node_type', 'unknown')
            })
        
        # Sauvegarder
        np.save(filepath.replace('.csv', '.npy'), adj_matrix.toarray())
        
        # Sauvegarder métadonnées et infos des nœuds séparément
        import json
        with open(filepath.replace('.csv', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(filepath.replace('.csv', '_nodes.json'), 'w') as f:
            json.dump(node_info, f, indent=2)
        
        print(f"💾 Matrice d'adjacence sauvegardée: {filepath}")
        print(f"💾 Métadonnées sauvegardées: {filepath.replace('.csv', '_metadata.json')}")
        print(f"💾 Infos nœuds sauvegardées: {filepath.replace('.csv', '_nodes.json')}")

def main():
    """Test du graph builder"""
    print("🍸 Cocktail GNN - Construction du graphe")
    print("=" * 50)
    
    # Construire le graphe
    builder = CocktailGraphBuilder("data/processed/cocktails_type_dataset.csv")  # Dataset des cocktails de type "Cocktail"
    
    # Nettoyage
    df_clean = builder.clean_ingredients()
    print("\n📋 Aperçu des données nettoyées:")
    print(df_clean[['name', 'ingredients_normalized']].head())
    
    # Construction du graphe
    builder.build_graph()
    
    # Statistiques
    stats = builder.get_graph_stats()
    print(f"\n📊 Statistiques du graphe:")
    for key, value in stats.items():
        if key != 'top_ingredients':
            print(f"   {key}: {value}")
    
    print(f"\n🔥 Top ingrédients les plus connectés:")
    for ingredient, degree in stats['top_ingredients']:
        print(f"   {ingredient}: {degree} cocktails")
    
    # Visualisation
    builder.visualize_graph_sample()
    
    # Sauvegarde
    builder.save_graph()
    builder.save_adjacency_matrix()
    
    print("\n✅ Graphe construit et sauvegardé !")

if __name__ == "__main__":
    main()