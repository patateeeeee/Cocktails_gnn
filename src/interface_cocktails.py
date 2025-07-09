"""
Interface Gradio pour le générateur de cocktails GNN
"""
import gradio as gr
import json
import plotly.graph_objects as go
import plotly.express as px
from models.cocktails_generator import CocktailGenerator
import pandas as pd
import numpy as np

class GradioCocktailDemo:
    """Gradio interface for the cocktail generator"""
    
    def __init__(self):
        print("🍸 Initializing generator...")
        try:
            self.generator = CocktailGenerator()
            print("✅ Generator ready!")
        except Exception as e:
            print(f"❌ Error loading generator: {e}")
            self.generator = None
    
    def get_available_spirits(self):
        """Returns only alcohol spirits (gin, vodka, rum, etc.)"""
        if self.generator:
            # Filter only alcohol spirits from ingredients
            spirits = []
            for ingredient in self.generator.ingredient_nodes:
                if any(spirit in ingredient.lower() for spirit in 
                      ['gin', 'vodka', 'rum', 'whiskey', 'whisky', 'tequila', 'brandy', 'cognac', 'bourbon']):
                    spirits.append(ingredient)
            return sorted(spirits)
        return []
    
    def generate_cocktail(self, base_ingredient, creativity, num_ingredients):
        """Generates a cocktail with given parameters"""
        if not self.generator:
            return "❌ Generator not available", "", ""
        
        try:
            # Validate spirit ingredient
            if base_ingredient not in self.get_available_spirits():
                available = ", ".join(self.get_available_spirits()[:10])
                return f"❌ Ingredient '{base_ingredient}' not found or not a spirit.\n\nAvailable spirits: {available}...", "", ""
            
            # Generate cocktail
            cocktail = self.generator.generate_cocktail_from_base(
                base_ingredient=base_ingredient,
                creativity=creativity,
                num_ingredients=num_ingredients
            )
            
            # Format results
            recipe_text = self._format_recipe(cocktail)
            instructions = cocktail['instructions']
            stats = self._format_stats(cocktail)
            
            return recipe_text, instructions, stats
            
        except Exception as e:
            return f"❌ Generation error: {str(e)}", "", ""
    
    def generate_cocktail_simple(self, base_ingredient, creativity, num_ingredients):
        """Génère un cocktail avec les paramètres donnés (version simplifiée sans profil gustatif)"""
        if not self.generator:
            return "❌ Générateur non disponible", "", "", ""
        
        try:
            # Validation de l'ingrédient
            if base_ingredient not in self.generator.ingredient_nodes:
                available = ", ".join(sorted(self.generator.ingredient_nodes)[:10])
                return f"❌ Ingrédient '{base_ingredient}' non trouvé.\n\nDisponibles: {available}...", "", "", ""
            
            # Génération du cocktail
            cocktail = self.generator.generate_cocktail_from_base(
                base_ingredient=base_ingredient,
                creativity=creativity,
                num_ingredients=num_ingredients
            )
            
            # Formatage des résultats
            recipe_text = self._format_recipe(cocktail)
            instructions = cocktail['instructions']
            stats = self._format_stats(cocktail)
            ingredient_list = self._format_ingredients(cocktail)
            
            return recipe_text, instructions, stats, ingredient_list
            
        except Exception as e:
            return f"❌ Erreur lors de la génération: {str(e)}", "", "", ""
    
    
    
    
    def generate_multiple_cocktails(self, num_cocktails, min_creativity, max_creativity):
        """Generate multiple cocktails with random creativity levels"""
        if not self.generator:
            return "❌ Generator not available"
        
        try:
            import random
            
            if min_creativity > max_creativity:
                return "❌ Min creativity must be <= Max creativity"
            
            results = "🍸 **Generated Cocktails:**\n\n"
            
            for i in range(int(num_cocktails)):
                creativity = random.uniform(min_creativity, max_creativity)
                cocktail = self.generator.generate_cocktail_from_base(
                    base_ingredient=random.choice(list(self.generator.ingredient_nodes)),
                    creativity=creativity,
                    num_ingredients=random.randint(3, 6)
                )
                
                results += f"## {i+1}. {cocktail['name']} (Creativity: {creativity:.2f})\n"
                results += f"**Ingredients:**\n"
                for ingredient in cocktail['ingredients']:
                    proportion = cocktail['proportions'].get(ingredient, "To taste")
                    results += f"• **{proportion}** {ingredient}\n"
                results += f"\n**Instructions:** {cocktail['instructions'][:100]}...\n\n"
                results += "---\n\n"
            
            return results
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_cocktail_from_available_ingredients(self, selected_ingredients, creativity, min_ingredients, max_ingredients):
        """Generate cocktail from selected ingredients"""
        if not self.generator:
            return "❌ Generator not available", "", ""
        
        try:
            if len(selected_ingredients) < min_ingredients:
                return f"❌ Select at least {min_ingredients} ingredients", "", ""
            
            cocktail = self.generator.generate_cocktail_with_limited_ingredients(
                available_ingredients=selected_ingredients,
                base_ingredient=None,
                creativity=creativity,
                num_ingredients=max_ingredients
            )
            
            recipe = f"# 🍸 {cocktail['name']}\n\n## Ingredients:\n"
            for ingredient in cocktail['ingredients']:
                proportion = cocktail['proportions'].get(ingredient, "To taste")
                recipe += f"• **{proportion}** {ingredient}\n"
            
            instructions = cocktail['instructions']
            
            stats = f"📊 **Stats**\n🎨 Creativity: {creativity:.2f}\n🥃 Ingredients: {len(cocktail['ingredients'])}"
            
            return recipe, instructions, stats
            
        except Exception as e:
            return f"❌ Error: {str(e)}", "", ""
    
    def _format_recipe(self, cocktail):
        """Formats recipe in markdown"""
        recipe = f"# 🍸 {cocktail['name']}\n\n"
        recipe += "## 📋 Ingredients\n"
        
        for ingredient in cocktail['ingredients']:
            proportion = cocktail['proportions'].get(ingredient, "To taste")
            recipe += f"• **{proportion}** {ingredient}\n"
        
        return recipe
    
    def _format_stats(self, cocktail):
        """Formats statistics"""
        novelty = self.generator.evaluate_cocktail_novelty(cocktail)
        creativity = cocktail.get('creativity_score', 0.5)
        
        stats = f"📊 **Performance Scores**\n\n"
        stats += f"🎨 **Creativity:** {creativity:.2f}/1.0\n"
        stats += f"✨ **Novelty:** {novelty:.2f}/1.0\n"
        stats += f"🥃 **Number of ingredients:** {len(cocktail['ingredients'])}\n"
        
        return stats
    
 
    
    
    
    
    def get_available_ingredients(self):
        """Retourne la liste des ingrédients disponibles"""
        if self.generator:
            return sorted(self.generator.ingredient_nodes)
        return []
    
    def get_available_cocktails(self):
        """Retourne la liste des cocktails disponibles"""
        if self.generator:
            return sorted(self.generator.cocktail_nodes)
        return []
    
    def get_ingredients_by_category(self):
        """Retourne les ingrédients organisés par catégorie"""
        if not self.generator:
            return {}
        
        categories = {
            'Spirits': [],
            'Liqueurs': [],
            'Wines & Champagne': [],
            'Juices & Citrus': [],
            'Syrups & Sweeteners': [],
            'Bitters & Aromatics': [],
            'Mixers': [],
            'Garnishes': [],
            'Others': []
        }
        
        for ingredient in sorted(self.generator.ingredient_nodes):
            ingredient_lower = ingredient.lower()
            
            # Spiritueux
            if any(spirit in ingredient_lower for spirit in 
                   ['gin', 'vodka', 'rum', 'whiskey', 'whisky', 'tequila', 'brandy', 'cognac', 'bourbon', 
                    'absinthe', 'cachaca']):
                categories['Spirits'].append(ingredient)
            
            # Liqueurs
            elif any(liq in ingredient_lower for liq in 
                    ['liqueur', 'cointreau', 'triple sec', 'amaretto', 'kahlua', 'chartreuse', 'curacao',
                     'baileys', 'grand marnier', 'passoa']):
                categories['Liqueurs'].append(ingredient)
            
            # Vins et Champagnes
            elif any(wine in ingredient_lower for wine in 
                    ['champagne', 'prosecco']):
                categories['Wines & Champagne'].append(ingredient)
            
            # Jus et agrumes
            elif any(juice in ingredient_lower for juice in 
                    ['juice', 'lemon', 'lime', 'orange', 'grapefruit', 'cranberry', 'pineapple']):
                categories['Juices & Citrus'].append(ingredient)
            
            # Sirops et sucrants
            elif any(syrup in ingredient_lower for syrup in 
                    ['syrup', 'grenadine', 'honey', 'agave', 'simple', 'elderflower']):
                categories['Syrups & Sweeteners'].append(ingredient)
            
            # Amers et aromatiques
            elif any(bitter in ingredient_lower for bitter in 
                    ['bitter', 'bitters', 'campari', 'aperol', 'vermouth']):
                categories['Bitters & Aromatics'].append(ingredient)
            
            # Mixers
            elif any(mixer in ingredient_lower for mixer in 
                    ['water', 'soda', 'tonic', 'ginger', 'cola']):
                categories['Mixers'].append(ingredient)
            
            # Garnitures
            elif any(garnish in ingredient_lower for garnish in 
                    ['cherry', 'olive', 'peel', 'mint', 'rosemary', 'blackberries']):
                categories['Garnishes'].append(ingredient)
            
            # Autres
            else:
                categories['Others'].append(ingredient)
        
        return categories
    
    def _format_recipe_with_availability(self, cocktail):
        """Formate la recette en montrant les ingrédients utilisés et non utilisés"""
        recipe = f"# 🍸 {cocktail['name']}\n\n"
        recipe += "## 📋 Ingrédients utilisés\n"
        
        for ingredient in cocktail['ingredients']:
            proportion = cocktail['proportions'].get(ingredient, "To taste")
            recipe += f"• **{proportion}** {ingredient}\n"
        
        if cocktail.get('unused_ingredients'):
            recipe += f"\n## 📦 Ingrédients disponibles non utilisés\n"
            for ingredient in cocktail['unused_ingredients']:
                recipe += f"• {ingredient}\n"
        
        return recipe
    
    def _format_stats_with_availability(self, cocktail):
        """Formate les statistiques avec info sur les ingrédients"""
        novelty = self.generator.evaluate_cocktail_novelty(cocktail)
        creativity = cocktail.get('creativity_score', 0.5)
        
        stats = f"📊 **Performance Scores**\n\n"
        stats += f"🎨 **Creativity:** {creativity:.2f}/1.0\n"
        stats += f"✨ **Novelty:** {novelty:.2f}/1.0\n"
        stats += f"🥃 **Ingrédients utilisés:** {len(cocktail['ingredients'])}\n"
        stats += f"📦 **Ingrédients disponibles:** {len(cocktail['available_ingredients'])}\n"
        stats += f"🔄 **Taux d'utilisation:** {len(cocktail['ingredients'])/len(cocktail['available_ingredients'])*100:.1f}%\n"
        
        return stats
    
    def _format_ingredients(self, cocktail):
        """Formate la liste des ingrédients"""
        ingredients_text = "🥃 **Ingrédients du cocktail:**\n\n"
        
        for ingredient in cocktail['ingredients']:
            proportion = cocktail['proportions'].get(ingredient, "To taste")
            ingredients_text += f"• {ingredient}: {proportion}\n"
        
        return ingredients_text
    
    def create_interface(self):
        """Create simple interface with 2 tabs"""
        
        with gr.Blocks(title="🍸 Cocktail Generator GNN") as demo:
            gr.Markdown("# 🍸 Cocktail Generator GNN")
            
            with gr.Tabs():
                # Tab 1: Multiple Generation
                with gr.TabItem("🥃 Classic Generation: MULTIPLE"):
                    gr.Markdown("## Generate multiple cocktails with random creativity")
                    
                    with gr.Row():
                        with gr.Column():
                            num_cocktails = gr.Slider(3, 10, value=5, step=1, label="Number of cocktails")
                            min_creativity = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="Min creativity")
                            max_creativity = gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="Max creativity")
                            generate_btn = gr.Button("🍸 Generate", variant="primary")
                        
                        with gr.Column():
                            output = gr.Markdown()
                    
                    generate_btn.click(
                        fn=self.generate_multiple_cocktails,
                        inputs=[num_cocktails, min_creativity, max_creativity],
                        outputs=[output]
                    )
                
                # Tab 2: Personalized with Ingredients
                with gr.TabItem("📦 Personalized with Ingredients"):
                    gr.Markdown("## Generate cocktail with your ingredients")
                    
                    with gr.Row():
                        with gr.Column():
                            creativity = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Creativity")
                            min_ing = gr.Slider(2, 6, value=3, step=1, label="Min ingredients")
                            max_ing = gr.Slider(3, 8, value=5, step=1, label="Max ingredients")
                            
                            # Simple ingredient selection
                            ingredients = gr.CheckboxGroup(
                                choices=self.get_available_ingredients(),
                                label="Select ingredients",
                                value=[]
                            )
                            
                            generate_btn2 = gr.Button("🍸 Generate", variant="primary")
                        
                        with gr.Column():
                            recipe_out = gr.Markdown()
                            instructions_out = gr.Markdown()
                            stats_out = gr.Markdown()
                    
                    generate_btn2.click(
                        fn=self.generate_cocktail_from_available_ingredients,
                        inputs=[ingredients, creativity, min_ing, max_ing],
                        outputs=[recipe_out, instructions_out, stats_out]
                    )
        
        return demo

def main():
    """Launch the interface"""
    print("🍸 Starting Cocktail Generator...")
    
    demo_app = GradioCocktailDemo()
    
    if demo_app.generator is None:
        print("❌ Generator not available")
        return
    
    demo = demo_app.create_interface()
    
    print("🚀 Interface ready!")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()

