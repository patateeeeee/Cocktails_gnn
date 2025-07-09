"""
Générateur de nouvelles recettes de cocktails basé sur les embeddings GNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class CocktailGenerator:
    """Générateur de nouvelles recettes utilisant les embeddings GNN"""
    
    def __init__(self, model_path: str = "data/processed/cocktail_gnn_model.pth"):
        # Charger le modèle entraîné
        self.device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Récupérer les métadonnées
        self.node_to_idx = checkpoint['node_to_idx']
        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}
        self.cocktail_nodes = checkpoint['cocktail_nodes']
        self.ingredient_nodes = checkpoint['ingredient_nodes']
        
        # Reconstruire le modèle
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from cocktails_gnn import CocktailGNN
        self.model = CocktailGNN(
            num_nodes=len(self.node_to_idx),
            input_dim=2,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Charger le graphe original pour les contraintes
        self.graph = nx.read_gml("data/processed/cocktail_graph.gml")
        
        # Calculer les embeddings une fois
        self._compute_embeddings()
        
        # Analyser les patterns existants
        self._analyze_cocktail_patterns()
        
        print(f"🍸 Générateur initialisé avec {len(self.ingredient_nodes)} ingrédients")
    
    def _compute_embeddings(self):
        """Calcule les embeddings de tous les nœuds"""
        # Préparer les données comme lors de l'entraînement
        from cocktails_gnn import CocktailDataset
        dataset = CocktailDataset()
        data = dataset.get_pytorch_geometric_data()
        
        with torch.no_grad():
            self.embeddings = self.model(data.x, data.edge_index)
        
        print("✅ Embeddings calculés")
    
    def _analyze_cocktail_patterns(self):
        """Analyse les patterns des cocktails existants"""
        self.patterns = {
            'ingredient_frequencies': defaultdict(int),
            'ingredient_combinations': defaultdict(int),
            'cocktail_sizes': [],
            'popular_bases': defaultdict(int),
            'modifier_types': defaultdict(int)
        }
        
        # Analyser chaque cocktail existant
        for cocktail in self.cocktail_nodes:
            ingredients = list(self.graph.neighbors(cocktail))
            self.patterns['cocktail_sizes'].append(len(ingredients))
            
            for ingredient in ingredients:
                self.patterns['ingredient_frequencies'][ingredient] += 1
                
                # Identifier les bases (spiritueux)
                if any(spirit in ingredient.lower() for spirit in 
                      ['gin', 'vodka', 'rum', 'whiskey', 'tequila', 'brandy']):
                    self.patterns['popular_bases'][ingredient] += 1
            
            # Analyser les combinaisons 2 à 2
            for i, ing1 in enumerate(ingredients):
                for ing2 in ingredients[i+1:]:
                    combo = tuple(sorted([ing1, ing2]))
                    self.patterns['ingredient_combinations'][combo] += 1
        
        # Stats
        avg_size = np.mean(self.patterns['cocktail_sizes'])
        print(f"📊 Patterns analysés: {avg_size:.1f} ingrédients en moyenne par cocktail")
    
    def find_similar_ingredients(self, ingredient: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Trouve les ingrédients les plus similaires basé sur les embeddings"""
        if ingredient not in self.node_to_idx:
            return []
        
        target_idx = self.node_to_idx[ingredient]
        target_emb = self.embeddings[target_idx]
        
        similarities = []
        for other_ingredient in self.ingredient_nodes:
            if other_ingredient != ingredient:
                other_idx = self.node_to_idx[other_ingredient]
                other_emb = self.embeddings[other_idx]
                
                # Similarité cosinus
                similarity = F.cosine_similarity(target_emb, other_emb, dim=0).item()
                similarities.append((other_ingredient, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def generate_cocktail_from_base(self, 
                                   base_ingredient: str,
                                   num_ingredients: int = None,
                                   creativity: float = 0.5) -> Dict:
        """Génère un cocktail à partir d'un ingrédient de base"""
        
        if base_ingredient not in self.ingredient_nodes:
            raise ValueError(f"Ingrédient '{base_ingredient}' non trouvé")
        
        # Taille du cocktail (basée sur les patterns ou spécifiée)
        if num_ingredients is None:
            avg_size = np.mean(self.patterns['cocktail_sizes'])
            num_ingredients = max(2, int(np.random.normal(avg_size, 1)))
        
        # Commencer avec l'ingrédient de base
        selected_ingredients = [base_ingredient]
        
        # Ajouter des ingrédients complémentaires
        for _ in range(num_ingredients - 1):
            next_ingredient = self._select_next_ingredient(
                selected_ingredients, creativity
            )
            if next_ingredient and next_ingredient not in selected_ingredients:
                selected_ingredients.append(next_ingredient)
        
        # Générer les proportions
        proportions = self._generate_proportions(selected_ingredients)
        
        # Créer la recette
        recipe = {
            'name': self._generate_name(selected_ingredients),
            'ingredients': selected_ingredients,
            'proportions': proportions,
            'instructions': self._generate_instructions(selected_ingredients, proportions),
            'estimated_taste': self._predict_taste_profile(selected_ingredients),
            'creativity_score': creativity
        }
        
        return recipe
    
    def _select_next_ingredient(self, current_ingredients: List[str], creativity: float) -> str:
        """Sélectionne le prochain ingrédient avec validation stricte des règles d'alcool"""
        
        # 1. VALIDATION STRICTE DES RÈGLES D'ALCOOL EN PREMIER
        valid_candidates = []
        
        for candidate in self.ingredient_nodes:
            if candidate not in current_ingredients:
                # Vérifier les règles d'alcool AVANT tout autre calcul
                if self._validate_strict_alcohol_rules(current_ingredients, candidate):
                    valid_candidates.append(candidate)
        
        if not valid_candidates:
            return None
        
        # 2. Calculer les scores seulement sur les candidats valides
        candidate_scores = {}
        
        # Embedding moyen des ingrédients actuels
        current_embeddings = []
        for ingredient in current_ingredients:
            idx = self.node_to_idx[ingredient]
            current_embeddings.append(self.embeddings[idx])
        
        avg_embedding = torch.stack(current_embeddings).mean(dim=0)
        
        for candidate in valid_candidates:
            # Score basé sur les vrais cocktails du dataset
            real_score = self._get_real_combination_score(current_ingredients, candidate)
            
            # Score basé sur la similarité des embeddings
            candidate_idx = self.node_to_idx[candidate]
            candidate_emb = self.embeddings[candidate_idx]
            similarity = F.cosine_similarity(avg_embedding, candidate_emb, dim=0).item()
            
            # Bonus pour les ingrédients complémentaires (non-alcoolisés)
            complement_bonus = self._get_complement_bonus(current_ingredients, candidate)
            
            # Score final équilibré

            final_score = (
                (1-creativity) * real_score +      # 40% basé sur les vrais cocktails
                creativity * similarity +      # 30% basé sur les embeddings
                0.2 * complement_bonus + # 20% pour l'équilibre
                0.1 * np.random.random() # 10% aléatoire
            )
            
            candidate_scores[candidate] = final_score
        
        # Sélection du meilleur candidat
        if candidate_scores:
            best_candidate = max(candidate_scores.items(), key=lambda x: x[1])[0]
            return best_candidate
        
        return None

    def _validate_strict_alcohol_rules(self, current_ingredients: List[str], candidate: str) -> bool:
        """Validation STRICTE des règles d'alcool selon la mixologie réelle"""
        
        # Analyser la composition actuelle
        current_composition = self._analyze_composition(current_ingredients)
        candidate_type = self._categorize_ingredient(candidate)
        
        # RÈGLE 1: Maximum 1 spiritueux de base
        if candidate_type == 'spirit':
            if current_composition['spirits'] >= 1:
                return False
        
        # RÈGLE 2: Maximum 1 liqueur principale
        elif candidate_type == 'liqueur':
            if current_composition['liqueurs'] >= 1:
                return False
        
        # RÈGLE 3: Si on a déjà 1 spiritueux, pas plus de 1 liqueur
        elif candidate_type == 'liqueur' and current_composition['spirits'] >= 1:
            if current_composition['liqueurs'] >= 1:
                return False
        
        # RÈGLE 4: Toujours privilégier les compléments non-alcoolisés
        non_alcohol_ratio = (
            len(current_ingredients) - 
            current_composition['spirits'] - 
            current_composition['liqueurs'] - 
            current_composition['fortified_wines']
        ) / max(len(current_ingredients), 1)
        
        # Si moins de 50% d'ingrédients non-alcoolisés, privilégier les non-alcoolisés
        if non_alcohol_ratio < 0.5 and candidate_type in ['spirit', 'liqueur']:
            return False
        
        return True

    def _analyze_composition(self, ingredients: List[str]) -> Dict[str, int]:
        """Analyse la composition alcoolique actuelle"""
        composition = {
            'spirits': 0,
            'liqueurs': 0,
            'fortified_wines': 0,
            'bitters': 0,
            'citrus': 0,
            'sweeteners': 0,
            'modifiers': 0
        }
        
        for ingredient in ingredients:
            category = self._categorize_ingredient(ingredient)
            if category == 'spirit':
                composition['spirits'] += 1
            elif category == 'liqueur':
                composition['liqueurs'] += 1
            elif category == 'fortified_wine':
                composition['fortified_wines'] += 1
            elif category == 'bitters':
                composition['bitters'] += 1
            elif category == 'citrus':
                composition['citrus'] += 1
            elif category == 'sweetener':
                composition['sweeteners'] += 1
            else:
                composition['modifiers'] += 1
        
        return composition

    def _get_complement_bonus(self, current_ingredients: List[str], candidate: str) -> float:
        """Bonus pour les ingrédients qui équilibrent la composition"""
        composition = self._analyze_composition(current_ingredients)
        candidate_type = self._categorize_ingredient(candidate)
        
        # Bonus pour équilibrer la composition
        if candidate_type == 'citrus' and composition['citrus'] == 0:
            return 0.8  # Cocktail sans agrume, bonus pour en ajouter
        elif candidate_type == 'sweetener' and composition['sweeteners'] == 0:
            return 0.6  # Cocktail sans sucrant, bonus modéré
        elif candidate_type == 'bitters' and composition['bitters'] == 0:
            return 0.4  # Bonus pour les amers
        elif candidate_type == 'modifier' and len(current_ingredients) >= 2:
            return 0.5  # Bonus pour les modificateurs (dilution)
        
        return 0.0

    def _generate_proportions(self, ingredients: List[str]) -> Dict[str, str]:
        """Génère des proportions RÉALISTES basées sur la mixologie classique"""
        proportions = {}
        
        # PROPORTIONS RÉALISTES PAR TYPE
        for ingredient in ingredients:
            category = self._categorize_ingredient(ingredient)
            ingredient_lower = ingredient.lower()
            
            if category == 'spirit':
                # Spiritueux de base : 45-60ml (standard)
                proportions[ingredient] = f"{np.random.randint(45, 61)} ml"
            
            elif category == 'liqueur':
                # Liqueurs : 15-30ml maximum
                proportions[ingredient] = f"{np.random.randint(15, 31)} ml"
            
            elif category == 'fortified_wine':
                # Vermouths : 15-25ml
                proportions[ingredient] = f"{np.random.randint(15, 26)} ml"
            
            elif category == 'sparkling_wine':
                # Champagne/Prosecco : 60-100ml (top up)
                proportions[ingredient] = f"{np.random.randint(60, 101)} ml"
            
            elif category == 'citrus':
                # Jus d'agrumes : 10-20ml
                if 'juice' in ingredient_lower:
                    proportions[ingredient] = f"{np.random.randint(10, 21)} ml"
                else:
                    proportions[ingredient] = f"{np.random.randint(8, 16)} ml"
            
            elif category == 'sweetener':
                # Sirops et édulcorants : proportions spécifiques
                if 'elderflower cordial' in ingredient_lower:
                    proportions[ingredient] = f"{np.random.randint(10, 21)} ml"
                elif 'honey' in ingredient_lower:
                    proportions[ingredient] = f"{np.random.randint(8, 16)} ml"
                elif 'sugar' in ingredient_lower and 'syrup' not in ingredient_lower:
                    proportions[ingredient] = f"{np.random.randint(1, 3)} tsp"
                else:
                    proportions[ingredient] = f"{np.random.randint(5, 16)} ml"
            
            elif category == 'bitters':
                # Amers : 2-4 dashes
                proportions[ingredient] = f"{np.random.randint(2, 5)} dashes"
            
            elif category == 'mixer':
                # Sodas et allongeurs
                if any(word in ingredient_lower for word in ['water', 'soda', 'tonic', 'lemonade']):
                    proportions[ingredient] = "top up"
                else:
                    proportions[ingredient] = f"{np.random.randint(60, 101)} ml"
            
            elif category == 'garnish':
                # Garnitures spécifiques
                if 'maraschino cherry' in ingredient_lower:
                    proportions[ingredient] = "1 cherry"
                elif 'cherry' in ingredient_lower:
                    proportions[ingredient] = "1-2 cherries"
                elif 'olive' in ingredient_lower:
                    proportions[ingredient] = "1-2 olives"
                elif 'peel' in ingredient_lower or 'twist' in ingredient_lower:
                    proportions[ingredient] = "1 twist"
                elif 'slice' in ingredient_lower:
                    proportions[ingredient] = "1 slice"
                elif 'wedge' in ingredient_lower:
                    proportions[ingredient] = "1 wedge"
                elif 'sprig' in ingredient_lower or 'rosemary' in ingredient_lower:
                    proportions[ingredient] = "1 sprig"
                elif 'mint' in ingredient_lower:
                    proportions[ingredient] = "6-8 leaves"
                elif any(fruit in ingredient_lower for fruit in ['blackberries', 'kiwi', 'mango']):
                    proportions[ingredient] = "garnish"
                else:
                    proportions[ingredient] = "garnish"
            
            elif category == 'texture':
                # Ingrédients texturants
                if 'egg white' in ingredient_lower:
                    proportions[ingredient] = "1 egg white"
                elif 'cream' in ingredient_lower:
                    proportions[ingredient] = f"{np.random.randint(10, 21)} ml"
                elif 'ice' in ingredient_lower:
                    proportions[ingredient] = "cubes"
                else:
                    proportions[ingredient] = f"{np.random.randint(10, 21)} ml"
            
            elif category == 'spice':
                # Épices
                if 'pepper' in ingredient_lower:
                    proportions[ingredient] = "pinch"
                else:
                    proportions[ingredient] = "pinch"
            
            else:
                # Autres modificateurs - cas spéciaux
                proportions[ingredient] = "to taste"
        
        return proportions
    
    def _get_real_proportions_for_ingredient(self, ingredient: str) -> List[float]:
        """Récupère les vraies proportions d'un ingrédient dans le dataset"""
        proportions = []
        
        for cocktail in self.cocktail_nodes:
            if self.graph.has_edge(cocktail, ingredient):
                edge_data = self.graph[cocktail][ingredient]
                weight = edge_data.get('weight', None)
                if weight and weight > 0:
                    proportions.append(weight)
        
        return proportions
    
    def _get_default_proportion(self, ingredient: str) -> str:
        """Proportions par défaut basées sur le type d'ingrédient"""
        ing_lower = ingredient.lower()
        
        if any(spirit in ing_lower for spirit in ['gin', 'vodka', 'rum', 'whiskey', 'tequila']):
            return f"{np.random.randint(45, 61)} ml"  # Base spirit
        elif any(modifier in ing_lower for modifier in ['vermouth', 'liqueur']):
            return f"{np.random.randint(15, 31)} ml"  # Modifier
        elif 'juice' in ing_lower or any(citrus in ing_lower for citrus in ['lemon', 'lime']):
            return f"{np.random.randint(15, 23)} ml"  # Citrus
        elif 'syrup' in ing_lower or 'grenadine' in ing_lower:
            return f"{np.random.randint(8, 16)} ml"  # Sweetener
        elif 'bitter' in ing_lower:
            return "2-3 dashes"
        else:
            return "a bit of"
        
        return proportions
    
    def _generate_name(self, ingredients: List[str]) -> str:
        """Génère un nom créatif pour le cocktail"""
        # Templates de noms basés sur les patterns
        templates = [
            "The {adjective} {noun}",
            "{ingredient} {noun}",
            "{adjective} {ingredient}",
            "{noun}'s {ingredient}",
            "{ingredient} & {noun}"
        ]
        
        adjectives = ["Golden", "Midnight", "Crimson", "Azure", "Velvet", "Crystal", 
                     "Electric", "Smoky", "Bright", "Dark", "Twisted", "Perfect"]
        
        nouns = ["Storm", "Sunset", "Dream", "Kiss", "Flight", "Garden", "Affair", 
                "Mystery", "Paradise", "Rebellion", "Symphony", "Enigma"]
        
        # Utiliser le premier ingrédient comme base parfois
        main_ingredient = ingredients[0].title()
        
        template = np.random.choice(templates)
        name = template.format(
            adjective=np.random.choice(adjectives),
            noun=np.random.choice(nouns),
            ingredient=main_ingredient
        )
        
        return name
    
    def _generate_instructions(self, ingredients: List[str], proportions: Dict[str, str]) -> str:
        """Génère des instructions de préparation"""
        
        # Analyser les types pour déterminer la méthode
        has_citrus = any('lemon' in ing or 'lime' in ing for ing in ingredients)
        has_cream = any('cream' in ing or 'milk' in ing for ing in ingredients)
        has_bitters = any('bitter' in ing for ing in ingredients)
        
        instructions = []
        
        if has_citrus or len(ingredients) > 3:
            instructions.append("Add all ingredients to a shaker filled with ice.")
            instructions.append("Shake vigorously for 10-15 seconds.")
            instructions.append("Double strain into a chilled coupe glass.")
        else:
            instructions.append("Add all ingredients to a mixing glass filled with ice.")
            instructions.append("Stir gently for 20-30 seconds.")
            instructions.append("Strain into a chilled rocks glass over a large ice cube.")
        
        # Garnish suggestions
        garnishes = ["Express lemon oils over the drink", "Garnish with an olive", 
                    "Add a cherry", "Garnish with orange peel"]
        instructions.append(np.random.choice(garnishes) + ".")
        
        return " ".join(instructions)
    
    def _predict_taste_profile(self, ingredients: List[str]) -> Dict[str, float]:
        """Prédit le profil gustatif du cocktail"""
        profile = {
            'sweet': 0.0,
            'sour': 0.0,
            'bitter': 0.0,
            'strong': 0.0,
            'fresh': 0.0
        }
        
        # Mapping ingrédients → profils (simplifié)
        taste_map = {
            'gin': {'bitter': 0.3, 'strong': 0.8},
            'vodka': {'strong': 0.7},
            'lime': {'sour': 0.9, 'fresh': 0.8},
            'lemon': {'sour': 0.8, 'fresh': 0.7},
            'grenadine': {'sweet': 0.9},
            'vermouth': {'bitter': 0.4, 'sweet': 0.3},
            'syrup': {'sweet': 0.8},
            'bitters': {'bitter': 1.0}
        }
        
        for ingredient in ingredients:
            for key_ingredient, tastes in taste_map.items():
                if key_ingredient in ingredient.lower():
                    for taste, value in tastes.items():
                        profile[taste] += value
        
        # Normaliser
        max_val = max(profile.values()) if max(profile.values()) > 0 else 1
        profile = {k: min(v/max_val, 1.0) for k, v in profile.items()}
        
        return profile
    
    def generate_multiple_cocktails(self, 
                                   num_cocktails: int = 5,
                                   base_ingredients: List[str] = None) -> List[Dict]:
        """Génère plusieurs cocktails avec différents niveaux de créativité"""
        
        if base_ingredients is None:
            # Utiliser les bases les plus populaires
            popular_bases = sorted(self.patterns['popular_bases'].items(), 
                                 key=lambda x: x[1], reverse=True)
            base_ingredients = [base for base, _ in popular_bases[:5]]
        
        cocktails = []
        creativity_levels = np.linspace(0.2, 0.8, num_cocktails)
        
        for i in range(num_cocktails):
            base = np.random.choice(base_ingredients)
            creativity = creativity_levels[i]
            
            try:
                cocktail = self.generate_cocktail_from_base(
                    base_ingredient=base,
                    creativity=creativity
                )
                cocktails.append(cocktail)
            except Exception as e:
                print(f"Erreur génération cocktail {i}: {e}")
        
        return cocktails
    
    def evaluate_cocktail_novelty(self, recipe: Dict) -> float:
        """Évalue la nouveauté d'un cocktail généré"""
        ingredients = set(recipe['ingredients'])
        
        # Calculer la similarité avec tous les cocktails existants
        max_similarity = 0.0
        
        for cocktail in self.cocktail_nodes:
            existing_ingredients = set(self.graph.neighbors(cocktail))
            
            # Coefficient de Jaccard
            intersection = len(ingredients & existing_ingredients)
            union = len(ingredients | existing_ingredients)
            similarity = intersection / union if union > 0 else 0.0
            
            max_similarity = max(max_similarity, similarity)
        
        # Nouveauté = 1 - similarité maximale
        novelty = 1.0 - max_similarity
        return novelty
    
    def _categorize_ingredients(self, ingredients: List[str]) -> Dict[str, List[str]]:
        """Catégorise les ingrédients par type"""
        categories = {
            'spirits': [],
            'liqueurs': [],
            'citrus': [],
            'syrups': [],
            'bitters': [],
            'modifiers': [],
            'garnish': []
        }
        
        for ingredient in ingredients:
            ing_lower = ingredient.lower()
            
            # Alcools forts (spiritueux)
            if any(spirit in ing_lower for spirit in 
                  ['gin', 'vodka', 'rum', 'whiskey', 'whisky', 'tequila', 'brandy', 'cognac', 'bourbon', 'scotch']):
                categories['spirits'].append(ingredient)
            
            # Liqueurs
            elif any(liq in ing_lower for liq in 
                    ['cointreau', 'triple sec', 'amaretto', 'kahlua', 'bailey', 'sambuca', 'chartreuse']):
                categories['liqueurs'].append(ingredient)
            
            # Agrumes
            elif any(citrus in ing_lower for citrus in 
                    ['lemon', 'lime', 'orange', 'grapefruit', 'citrus']):
                categories['citrus'].append(ingredient)
            
            # Sirops
            elif any(syrup in ing_lower for syrup in 
                    ['syrup', 'grenadine', 'honey', 'agave', 'simple']):
                categories['syrups'].append(ingredient)
            
            # Bitters
            elif 'bitter' in ing_lower:
                categories['bitters'].append(ingredient)
            
            # Modificateurs (vermouths, etc.)
            elif any(mod in ing_lower for mod in 
                    ['vermouth', 'aperol', 'campari', 'sherry', 'port']):
                categories['modifiers'].append(ingredient)
            
            # Garnish et divers
            else:
                categories['garnish'].append(ingredient)
        
        return categories
    
    def _categorize_ingredient(self, ingredient: str) -> str:
        """Catégorise un ingrédient selon son type basé sur TOUS les ingrédients du dataset"""
        ingredient_lower = ingredient.lower()
        
        # Spiritueux (alcools forts de base)
        if any(spirit in ingredient_lower for spirit in 
               ['gin', 'vodka', 'rum', 'whiskey', 'whisky', 'tequila', 'brandy', 'cognac', 'bourbon',
                'rye whiskey', '151 proof rum', 'dark rum', 'white rum', 'light rum', 'cachaca', 
                'pisco', 'absinthe', 'irish whiskey']):
            return 'spirit'
        
        # Liqueurs (alcools sucrés/aromatisés)
        elif any(liqueur in ingredient_lower for liqueur in 
                ['cointreau', 'triple sec', 'amaretto', 'kahlua', 'baileys', 'grand marnier', 
                 'chambord', 'frangelico', 'drambuie', 'chartreuse', 'benedictine', 'licor',
                 'liqueur', 'schnapps', 'sambuca', 'ouzo', 'maraschino liqueur', 'green chartreuse',
                 'blue curacao', 'baileys irish cream', 'galliano', 'coconut liqueur', 'passoa']):
            return 'liqueur'
        
        # Vermouths et vins fortifiés
        elif any(wine in ingredient_lower for wine in 
                ['vermouth', 'vermouth sweet', 'vermouth dry', 'sherry', 'port', 'madeira', 
                 'marsala', 'lillet blanc']):
            return 'fortified_wine'
        
        # Vins pétillants et champagne
        elif any(sparkling in ingredient_lower for sparkling in 
                ['champagne', 'prosecco']):
            return 'sparkling_wine'
        
        # Amers et bitters
        elif any(bitter in ingredient_lower for bitter in 
                ['bitter', 'bitters', 'orange bitters', 'angostura', 'campari', 'aperol', 'fernet']):
            return 'bitters'
        
        # Jus et acidulants (agrumes et fruits)
        elif any(juice in ingredient_lower for juice in 
                ['juice', 'lemon', 'lime', 'orange', 'grapefruit', 'cranberry', 'pineapple juice',
                 'orange juice', 'cranberry juice', 'grapefruit juice', 'passion fruit juice']):
            return 'citrus'
        
        # Sirops et sucrants
        elif any(syrup in ingredient_lower for syrup in 
                ['syrup', 'sugar syrup', 'grenadine', 'honey', 'agave', 'simple', 'sugar',
                 'elderflower cordial']):
            return 'sweetener'
        
        # Sodas et allongeurs
        elif any(mixer in ingredient_lower for mixer in 
                ['soda water', 'tonic water', 'club soda', 'water', 'lemonade']):
            return 'mixer'
        
        # Garnitures et décorations
        elif any(garnish in ingredient_lower for garnish in 
                ['maraschino cherry', 'olive', 'orange peel', 'lemon peel', 'cherry', 'peel',
                 'blackberries', 'kiwi', 'mango', 'rosemary', 'mint']):
            return 'garnish'
        
        # Ingrédients texturants/protéinés
        elif any(texture in ingredient_lower for texture in 
                ['egg white', 'cream', 'ice']):
            return 'texture'
        
        # Épices et assaisonnements
        elif any(spice in ingredient_lower for spice in 
                ['pepper']):
            return 'spice'
        
        # Autres modificateurs (catch-all pour les ingrédients non classés)
        else:
            return 'modifier'
    
    def _validate_alcohol_rules(self, current_ingredients: List[str], candidate: str) -> bool:
        """Valide les règles de limite d'alcool"""
        # Compter les alcools actuels
        current_spirits = 0
        current_liqueurs = 0
        current_fortified = 0
        
        for ingredient in current_ingredients:
            category = self._categorize_ingredient(ingredient)
            if category == 'spirit':
                current_spirits += 1
            elif category == 'liqueur':
                current_liqueurs += 1
            elif category == 'fortified_wine':
                current_fortified += 1
        
        # Catégoriser le candidat
        candidate_category = self._categorize_ingredient(candidate)
        
        # Règles de validation
        if candidate_category == 'spirit':
            # Maximum 2 spiritueux
            if current_spirits >= 2:
                return False
            # Si on a déjà 1 spiritueux + 1 liqueur, pas de 2e spiritueux
            if current_spirits >= 1 and current_liqueurs >= 1:
                return False
        
        elif candidate_category == 'liqueur':
            # Maximum 2 liqueurs
            if current_liqueurs >= 2:
                return False
            # Si on a déjà 1 spiritueux + 1 liqueur, pas de 2e liqueur
            if current_spirits >= 1 and current_liqueurs >= 1:
                return False
        
        elif candidate_category == 'fortified_wine':
            # Maximum 1 vin fortifié
            if current_fortified >= 1:
                return False
        
        return True
    
  
    def _get_real_combination_score(self, current_ingredients: List[str], candidate: str) -> float:
        """Score basé sur la fréquence de cette combinaison dans les vrais cocktails"""
        current_set = set(current_ingredients + [candidate])
        
        # Compter combien de vrais cocktails contiennent cette combinaison ou une partie
        match_count = 0
        total_cocktails = len(self.cocktail_nodes)
        
        for cocktail in self.cocktail_nodes:
            cocktail_ingredients = set(self.graph.neighbors(cocktail))
            
            # Score basé sur le nombre d'ingrédients en commun
            intersection = len(current_set & cocktail_ingredients)
            if intersection >= len(current_set) * 0.5:  # Au moins 50% en commun
                match_count += intersection / len(current_set)
        
        # Normaliser le score
        return min(match_count / total_cocktails, 1.0)
    
    def _analyze_real_proportions(self) -> Dict[str, Dict[str, float]]:
        """Analyse les vraies proportions du dataset pour chaque type d'ingrédient"""
        proportions_analysis = {
            'spirits': [],
            'modifiers': [],
            'citrus': [],
            'others': []
        }
        
        for cocktail in self.cocktail_nodes:
            for ingredient in self.graph.neighbors(cocktail):
                edge_data = self.graph[cocktail][ingredient]
                weight = edge_data.get('weight', 30.0)  # ml
                
                # Catégoriser et enregistrer le poids
                if any(spirit in ingredient.lower() for spirit in 
                      ['gin', 'vodka', 'rum', 'whiskey', 'tequila', 'brandy']):
                    proportions_analysis['spirits'].append(weight)
                elif any(mod in ingredient.lower() for mod in 
                        ['vermouth', 'liqueur', 'syrup']):
                    proportions_analysis['modifiers'].append(weight)
                elif any(citrus in ingredient.lower() for citrus in 
                        ['lemon', 'lime', 'orange']):
                    proportions_analysis['citrus'].append(weight)
                else:
                    proportions_analysis['others'].append(weight)
        
        # Calculer les statistiques
        stats = {}
        for category, weights in proportions_analysis.items():
            if weights:
                stats[category] = {
                    'mean': np.mean(weights),
                    'median': np.median(weights),
                    'min': np.min(weights),
                    'max': np.max(weights)
                }
        
        return stats
    
def main():
    """Test du générateur de cocktails"""
    print("🍸 Cocktail Generator - Test de génération")
    print("=" * 50)
    
    # Initialiser le générateur
    generator = CocktailGenerator()
    
    # Test 1: Générer des cocktails à partir de différentes bases
    print("\n🥃 Génération de cocktails par base:")
    bases = ['gin', 'vodka', 'rum']
    
    for base in bases:
        if base in generator.ingredient_nodes:
            print(f"\n--- Cocktail à base de {base.upper()} ---")
            cocktail = generator.generate_cocktail_from_base(base, creativity=0.5)
            
            print(f"🍸 {cocktail['name']}")
            print("Ingrédients:")
            for ingredient, proportion in zip(cocktail['ingredients'], 
                                            cocktail['proportions'].values()):
                print(f"  - {proportion} {ingredient}")
            
            print(f"Instructions: {cocktail['instructions']}")
            print(f"Profil gustatif: {cocktail['estimated_taste']}")
            print(f"Nouveauté: {generator.evaluate_cocktail_novelty(cocktail):.2f}")
    
    # Test 2: Batch de cocktails créatifs
    print(f"\n🎨 Génération de 5 cocktails créatifs:")
    creative_cocktails = generator.generate_multiple_cocktails(5)
    
    for i, cocktail in enumerate(creative_cocktails, 1):
        print(f"\n{i}. {cocktail['name']}")
        print(f"   Ingrédients: {', '.join(cocktail['ingredients'])}")
        print(f"   Créativité: {cocktail['creativity_score']:.2f}")
        print(f"   Nouveauté: {generator.evaluate_cocktail_novelty(cocktail):.2f}")
    
    # Test 3: Analyse de similarité d'ingrédients
    print(f"\n🔍 Ingrédients similaires à 'gin':")
    similar = generator.find_similar_ingredients('gin', top_k=5)
    for ingredient, similarity in similar:
        print(f"  {ingredient}: {similarity:.3f}")
    
    print("\n✅ Tests de génération terminés !")

if __name__ == "__main__":
    main()