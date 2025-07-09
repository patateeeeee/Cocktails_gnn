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
    
    def find_similar_ingredients(self, ingredient, top_k):
        """Trouve des ingrédients similaires"""
        if not self.generator:
            return "❌ Générateur non disponible"
        
        try:
            if ingredient not in self.generator.ingredient_nodes:
                return f"❌ Ingrédient '{ingredient}' non trouvé"
            
            similar = self.generator.find_similar_ingredients(ingredient, top_k=top_k)
            
            # Créer un graphique de similarité
            ingredients = [item[0] for item in similar]
            similarities = [item[1] for item in similar]
            
            fig = go.Figure(data=go.Bar(
                x=similarities,
                y=ingredients,
                orientation='h',
                marker_color='rgba(55, 128, 191, 0.7)',
                marker_line_color='rgba(55, 128, 191, 1.0)',
                marker_line_width=2
            ))
            
            fig.update_layout(
                title=f"🔍 Ingrédients similaires à '{ingredient}'",
                xaxis_title="Similarité",
                yaxis_title="Ingrédients",
                height=400,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            return f"❌ Erreur: {str(e)}"
    
    def analyze_cocktail(self, cocktail_name):
        """Analyse un cocktail existant"""
        if not self.generator:
            return "❌ Générateur non disponible", ""
        
        try:
            # Recherche flexible du cocktail
            found_cocktail = None
            for cocktail in self.generator.cocktail_nodes:
                if cocktail_name.lower() in cocktail.lower():
                    found_cocktail = cocktail
                    break
            
            if not found_cocktail:
                available = ", ".join(sorted(self.generator.cocktail_nodes)[:10])
                return f"❌ Cocktail '{cocktail_name}' non trouvé.\n\nDisponibles: {available}...", ""
            
            # Analyser le cocktail
            ingredients = list(self.generator.graph.neighbors(found_cocktail))
            
            analysis = f"📊 **Analyse de '{found_cocktail}'**\n\n"
            analysis += f"🥃 **Ingrédients ({len(ingredients)}):**\n"
            
            for ingredient in ingredients:
                edge_data = self.generator.graph[found_cocktail][ingredient]
                measurement = edge_data.get('measurement_raw', 'N/A')
                analysis += f"  • {ingredient} ({measurement})\n"
            
            # Cocktails similaires
            analysis += f"\n🔍 **Cocktails similaires:**\n"
            similar_cocktails = self._find_similar_cocktails(found_cocktail, ingredients)
            
            for cocktail, similarity in similar_cocktails[:5]:
                stars = "⭐" * int(similarity * 5)
                analysis += f"  • {cocktail} {stars} ({similarity:.2f})\n"
            
            # Créer un graphique des ingrédients
            ingredient_chart = self._create_ingredient_chart(found_cocktail, ingredients)
            
            return analysis, ingredient_chart
            
        except Exception as e:
            return f"❌ Erreur: {str(e)}", ""
    
    def generate_multiple_cocktails(self, num_cocktails, creativity_range):
        """Génère plusieurs cocktails avec différents niveaux de créativité"""
        if not self.generator:
            return "❌ Générateur non disponible"
        
        try:
            # Générer les cocktails
            cocktails = self.generator.generate_multiple_cocktails(num_cocktails)
            
            # Formatter les résultats
            results = "🍸 **Cocktails générés:**\n\n"
            
            for i, cocktail in enumerate(cocktails, 1):
                novelty = self.generator.evaluate_cocktail_novelty(cocktail)
                creativity = cocktail.get('creativity_score', 0.5)
                
                results += f"## {i}. {cocktail['name']}\n"
                results += f"**Ingrédients:** {', '.join(cocktail['ingredients'])}\n"
                results += f"**Créativité:** {creativity:.2f}/1.0 | **Nouveauté:** {novelty:.2f}/1.0\n\n"
                
                # Instructions courtes
                instructions = cocktail['instructions']
                if len(instructions) > 100:
                    instructions = instructions[:100] + "..."
                results += f"*{instructions}*\n\n"
                results += "---\n\n"
            
            return results
            
        except Exception as e:
            return f"❌ Erreur: {str(e)}"
    
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
    
    def _create_taste_chart(self, taste_profile):
        """Crée un graphique radar du profil gustatif"""
        categories = list(taste_profile.keys())
        values = list(taste_profile.values())
        
        # Ajouter le premier point à la fin pour fermer le radar
        categories += [categories[0]]
        values += [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Profil gustatif',
            line_color='rgba(255, 99, 71, 0.8)',
            fillcolor='rgba(255, 99, 71, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="👅 Profil gustatif prédit",
            height=400
        )
        
        return fig
    
    def _create_ingredient_chart(self, cocktail_name, ingredients):
        """Crée un graphique des ingrédients d'un cocktail"""
        # Analyser les types d'ingrédients
        types = []
        for ingredient in ingredients:
            if any(spirit in ingredient.lower() for spirit in ['gin', 'vodka', 'rum', 'whiskey', 'tequila']):
                types.append('Spiritueux')
            elif any(mod in ingredient.lower() for mod in ['vermouth', 'liqueur', 'syrup']):
                types.append('Modificateur')
            elif any(cit in ingredient.lower() for cit in ['lemon', 'lime', 'orange']):
                types.append('Agrume')
            else:
                types.append('Autre')
        
        # Compter les types
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Créer le graphique
        fig = go.Figure(data=go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.3,
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ))
        
        fig.update_layout(
            title=f"🥃 Composition de '{cocktail_name}'",
            height=400
        )
        
        return fig
    
    def _find_similar_cocktails(self, target_cocktail, target_ingredients):
        """Trouve des cocktails similaires"""
        similar_cocktails = []
        target_ingredients = set(target_ingredients)
        
        for other_cocktail in self.generator.cocktail_nodes:
            if other_cocktail != target_cocktail:
                other_ingredients = set(self.generator.graph.neighbors(other_cocktail))
                
                intersection = len(target_ingredients & other_ingredients)
                union = len(target_ingredients | other_ingredients)
                similarity = intersection / union if union > 0 else 0.0
                
                if similarity > 0.1:
                    similar_cocktails.append((other_cocktail, similarity))
        
        similar_cocktails.sort(key=lambda x: x[1], reverse=True)
        return similar_cocktails
    
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

def create_gradio_interface():
    """Creates the Gradio interface"""
    demo_app = GradioCocktailDemo()
    
    # Custom CSS for styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24) !important;
        border: none !important;
    }
    .gr-box {
        border-radius: 15px !important;
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🍸 Cocktail GNN Generator") as app:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #ff6b6b, #ee5a24); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 3em;">🍸 Cocktail GNN</h1>
            <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">AI-powered cocktail recipe generator with Graph Neural Network</p>
        </div>
        """)
        
        # Tabs for different functionalities
        with gr.Tabs():
            
            # Tab 1: Single generation
            with gr.TabItem("🍸 Generate a cocktail"):
                with gr.Row():
                    with gr.Column(scale=1):
                        base_ingredient = gr.Dropdown(
                            choices=demo_app.get_available_spirits(),
                            label="🥃 Base spirit",
                            info="Choose your main spirit (alcohol base required)"
                        )
                        creativity = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="🎨 Creativity level",
                            info="0.0 = Classic, 1.0 = Very creative"
                        )
                        num_ingredients = gr.Slider(
                            minimum=2,
                            maximum=8,
                            value=4,
                            step=1,
                            label="📊 Number of ingredients"
                        )
                        generate_btn = gr.Button("🔮 Generate cocktail", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        recipe_output = gr.Markdown(label="📋 Recipe")
                
                with gr.Row():
                    with gr.Column():
                        instructions_output = gr.Textbox(label="📝 Instructions", lines=4)
                    
                    with gr.Column():
                        stats_output = gr.Markdown(label="📊 Statistics")
                
                generate_btn.click(
                    fn=demo_app.generate_cocktail,
                    inputs=[base_ingredient, creativity, num_ingredients],
                    outputs=[recipe_output, instructions_output, stats_output]
                )
            
            # Tab 2: Multiple generation
            with gr.TabItem("🍸✨ Generate multiple cocktails"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Parameters for multiple generation
                        multiple_base = gr.Dropdown(
                            choices=["Random"] + demo_app.get_available_spirits(),
                            value="Random",
                            label="🥃 Base spirit (optional)",
                            info="Leave 'Random' to vary base spirits"
                        )
                        num_cocktails = gr.Slider(
                            minimum=2,
                            maximum=10,
                            value=5,
                            step=1,
                            label="🔢 Number of cocktails",
                            info="How many cocktails to generate"
                        )
                        creativity_min = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="🎨 Minimum creativity"
                        )
                        creativity_max = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.1,
                            label="🎨 Maximum creativity"
                        )
                        generate_multiple_btn = gr.Button("🔮✨ Generate cocktails", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        multiple_output = gr.Markdown(
                            label="🍸 Generated cocktails", 
                            value="Click 'Generate cocktails' to start!",
                            max_height=600
                        )
                
                # Function to generate multiple cocktails
                def generate_multiple_cocktails_wrapper(base, num_cocktails, creativity_min, creativity_max):
                    if not demo_app.generator:
                        return "❌ Generator not available"
                    
                    try:
                        results = "# 🍸 Generated Cocktails\n\n"
                        successful_generations = 0
                        
                        for i in range(num_cocktails):
                            try:
                                # Choose random base if necessary
                                if base == "Random":
                                    import random
                                    current_base = random.choice(demo_app.get_available_spirits())
                                else:
                                    current_base = base
                                
                                # Random creativity level in range
                                import random
                                current_creativity = random.uniform(creativity_min, creativity_max)
                                
                                # Random number of ingredients
                                current_num_ingredients = random.randint(3, 6)
                                
                                # Generate cocktail
                                cocktail = demo_app.generator.generate_cocktail_from_base(
                                    base_ingredient=current_base,
                                    creativity=current_creativity,
                                    num_ingredients=current_num_ingredients
                                )
                                
                                # Format result
                                novelty = demo_app.generator.evaluate_cocktail_novelty(cocktail)
                                
                                results += f"## {successful_generations + 1}. 🍸 {cocktail['name']}\n\n"
                                results += f"**🥃 Base:** {current_base} | **🎨 Creativity:** {current_creativity:.2f} | **✨ Novelty:** {novelty:.2f}\n\n"
                                
                                results += "**📋 Ingredients:**\n"
                                for ingredient in cocktail['ingredients']:
                                    proportion = cocktail['proportions'].get(ingredient, "To taste")
                                    results += f"• {proportion} {ingredient}\n"
                                
                                results += f"\n**📝 Instructions:** {cocktail['instructions'][:150]}...\n\n"
                                results += "---\n\n"
                                
                                successful_generations += 1
                                
                            except Exception as e:
                                results += f"❌ Error for cocktail {i+1}: {str(e)}\n\n"
                        
                        if successful_generations == 0:
                            return "❌ No cocktails could be generated. Check parameters."
                        
                        results += f"✅ **{successful_generations}/{num_cocktails} cocktails generated successfully!**"
                        return results
                        
                    except Exception as e:
                        return f"❌ Error during multiple generation: {str(e)}"
                
                generate_multiple_btn.click(
                    fn=generate_multiple_cocktails_wrapper,
                    inputs=[multiple_base, num_cocktails, creativity_min, creativity_max],
                    outputs=[multiple_output]
                )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; opacity: 0.7;">
            <p>🤖 Powered by GraphSAGE GNN | 🧠 Trained on 97 cocktails</p>
            <p>Intelligent cocktail recipe generator 🍸</p>
        </div>
        """)
    
    return app

def main():
    """Launches the Gradio interface"""
    print("🚀 Launching Gradio interface...")
    
    try:
        app = create_gradio_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Creates a public link
            debug=True
        )
    except Exception as e:
        print(f"❌ Launch error: {e}")

if __name__ == "__main__":
    main()
    main()