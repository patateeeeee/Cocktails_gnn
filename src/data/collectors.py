"""
Data collectors pour récupérer les données de cocktails depuis différentes sources
"""
import requests
import pandas as pd
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TheCocktailDBCollector:
    """Collecteur pour l'API TheCocktailDB"""
    
    BASE_URL = "https://www.thecocktaildb.com/api/json/v1/1"
    
    def __init__(self, save_path: str = "data/raw"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def get_cocktails_by_letter(self, letter: str) -> List[Dict]:
        """Récupère tous les cocktails commençant par une lettre"""
        url = f"{self.BASE_URL}/search.php?f={letter}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data['drinks']:
                logger.info(f"Trouvé {len(data['drinks'])} cocktails pour la lettre '{letter}'")
                return data['drinks']
            else:
                logger.info(f"Aucun cocktail trouvé pour la lettre '{letter}'")
                return []
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération pour '{letter}': {e}")
            return []
    
    def get_all_cocktails(self, delay: float = 0.1) -> List[Dict]:
        """Récupère tous les cocktails de A à Z"""
        all_cocktails = []
        
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            cocktails = self.get_cocktails_by_letter(letter)
            all_cocktails.extend(cocktails)
            
            # Rate limiting pour être sympa avec l'API
            time.sleep(delay)
            
        logger.info(f"Total: {len(all_cocktails)} cocktails récupérés")
        return all_cocktails
    
    def search_cocktail(self, name: str) -> List[Dict]:
        """Recherche un cocktail par nom"""
        url = f"{self.BASE_URL}/search.php?s={name}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            return data['drinks'] if data['drinks'] else []
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de '{name}': {e}")
            return []
    
    def save_cocktails(self, cocktails: List[Dict], filename: str = "thecocktaildb_raw.json"):
        """Sauvegarde les cocktails en JSON"""
        filepath = self.save_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cocktails, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Cocktails sauvegardés dans {filepath}")

class CocktailDataProcessor:
    """Processeur pour nettoyer et standardiser les données de cocktails"""
    
    def __init__(self, raw_data_path: str = "data/raw"):
        self.raw_path = Path(raw_data_path)
        self.processed_path = Path("data/processed")
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def clean_thecocktaildb_data(self, data: List[Dict]) -> pd.DataFrame:
        """Nettoie et structure les données de TheCocktailDB"""
        cleaned_cocktails = []
        
        for drink in data:
            cocktail = {
                'id': drink.get('idDrink'),
                'name': drink.get('strDrink'),
                'category': drink.get('strCategory'),
                'alcoholic': drink.get('strAlcoholic'),
                'glass': drink.get('strGlass'),
                'instructions': drink.get('strInstructions'),
                'image_url': drink.get('strDrinkThumb'),
                'ingredients': [],
                'measurements': []
            }
            
            # Extraction des ingrédients et mesures (jusqu'à 15 possibles)
            for i in range(1, 16):
                ingredient = drink.get(f'strIngredient{i}')
                measurement = drink.get(f'strMeasure{i}')
                
                if ingredient and ingredient.strip():
                    cocktail['ingredients'].append(ingredient.strip().lower())
                    cocktail['measurements'].append(measurement.strip() if measurement else "")
            
            if cocktail['ingredients']:  # Seulement si on a des ingrédients
                cleaned_cocktails.append(cocktail)
        
        df = pd.DataFrame(cleaned_cocktails)
        logger.info(f"Nettoyé {len(df)} cocktails")
        return df
    
    def extract_ingredients_df(self, cocktails_df: pd.DataFrame) -> pd.DataFrame:
        """Extrait une liste unique d'ingrédients avec leurs stats"""
        all_ingredients = []
        
        for _, cocktail in cocktails_df.iterrows():
            for ingredient in cocktail['ingredients']:
                all_ingredients.append(ingredient)
        
        ingredients_df = pd.DataFrame(all_ingredients, columns=['name'])
        ingredients_stats = ingredients_df.groupby('name').size().reset_index(name='frequency')
        ingredients_stats = ingredients_stats.sort_values('frequency', ascending=False)
        
        logger.info(f"Trouvé {len(ingredients_stats)} ingrédients uniques")
        return ingredients_stats
    
    def save_processed_data(self, cocktails_df: pd.DataFrame, ingredients_df: pd.DataFrame):
        """Sauvegarde les données nettoyées"""
        cocktails_path = self.processed_path / "cocktails_cleaned.csv"
        ingredients_path = self.processed_path / "ingredients_stats.csv"
        
        cocktails_df.to_csv(cocktails_path, index=False)
        ingredients_df.to_csv(ingredients_path, index=False)
        
        logger.info(f"Données sauvegardées dans {self.processed_path}")

def main():
    """Fonction principale pour récupérer TOUS les cocktails"""
    print("🍸 Cocktail GNN - Data Collection COMPLÈTE")
    print("=" * 50)
    
    # Récupération de TOUS les cocktails de A à Z
    collector = TheCocktailDBCollector()
    
    print("🔄 Récupération de tous les cocktails de A à Z...")
    print("⏳ Cela peut prendre quelques minutes...")
    
    all_cocktails = collector.get_all_cocktails(delay=0.1)
    
    print(f"\n✅ Total récupéré: {len(all_cocktails)} cocktails")
    
    # Sauvegarde de tous les cocktails
    collector.save_cocktails(all_cocktails, "all_cocktails_complete.json")
    
    # Nettoyage des données
    processor = CocktailDataProcessor()
    cocktails_df = processor.clean_thecocktaildb_data(all_cocktails)
    ingredients_df = processor.extract_ingredients_df(cocktails_df)
    
    # Affichage des stats
    print(f"\nCocktails nettoyés: {len(cocktails_df)}")
    print(f"Ingrédients uniques: {len(ingredients_df)}")
    print("\nTop 10 ingrédients:")
    print(ingredients_df.head(10))
    
    # Sauvegarde
    processor.save_processed_data(cocktails_df, ingredients_df)
    
    print("\n✅ Data collection COMPLÈTE terminée !")
    print(f"📊 {len(cocktails_df)} cocktails traités et sauvegardés")
    print(f"🧪 {len(ingredients_df)} ingrédients uniques identifiés")

if __name__ == "__main__":
    main()