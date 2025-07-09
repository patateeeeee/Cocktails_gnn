# 🍸 AI Cocktail Generator with Graph Neural Networks

**🚀 [Try the Live Demo on Hugging Face](https://huggingface.co/spaces/Paataatee/Cocktails_GNN)**

An AI-powered cocktail recipe generator that uses Graph Neural Networks (GNN) to create new cocktail recipes. The system can generate cocktails either randomly or with specific ingredients you provide, using learned embeddings to intelligently select compatible ingredients.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## � Generation Modes

### 1. Classic Generation: MULTIPLE
Generate completely random cocktails with adjustable creativity levels. You can specify minimum and maximum creativity bounds, and the system will create multiple cocktails with random creativity values in that range.

### 2. Personalized with Ingredients
Select your available ingredients and let the AI create cocktails using those ingredients. The GNN embeddings help select the most compatible combinations and ensure realistic proportions.

## 🧠 How It Works

### Creativity as Temperature Parameter
The **creativity** parameter acts as a temperature control that determines how much risk the model takes when selecting ingredients:
- **Low creativity (0.1-0.4)**: Conservative choices, classic combinations
- **Medium creativity (0.5-0.7)**: Balanced approach, some experimentation
- **High creativity (0.8-1.0)**: Bold choices, experimental combinations

### GNN-Based Ingredient Selection
The system uses Graph Neural Network embeddings to make intelligent ingredient choices:
- **GraphSAGE** learns 128-dimensional embeddings for all ingredients
- **Cosine similarity** in embedding space identifies compatible ingredients
- **Real cocktail relationships** from training data guide selections
- **Mixology rules** ensure realistic cocktail composition (max 1 spirit + 1 liqueur)
- **Smart proportions** generate realistic measurements based on ingredient types

## 🏗️ Architecture

### Dataset
- **97 cocktail recipes** from TheCocktailDB API
- **75 unique ingredients** with frequency analysis
- **324 cocktail-ingredient connections** in bipartite graph

### Graph Structure
```
Bipartite Graph:
├── Cocktail Nodes (97)
├── Ingredient Nodes (75)
└── Weighted Edges (324) - with real proportions
```

### Neural Network
- **GraphSAGE** architecture for node embeddings
- **128-dimensional** ingredient embeddings
- **3-layer** message passing
- **Embedding-based** ingredient selection

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cocktail_gnn.git
cd cocktail_gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Launch the Interactive Interface
```bash
python src/interface_cocktails.py
```

#### Generate Cocktails Programmatically
```python
from models.cocktails_generator import CocktailGenerator

generator = CocktailGenerator()

# Generate with random creativity in a range
cocktails = generator.generate_multiple_cocktails_with_creativity_bounds(
    num_cocktails=5,
    min_creativity=0.3,
    max_creativity=0.8
)

# Generate with specific available ingredients
available_ingredients = ['gin', 'lemon juice', 'sugar syrup', 'tonic water']
cocktail = generator.generate_cocktail_with_limited_ingredients(
    available_ingredients=available_ingredients,
    creativity=0.6
)

print(f"🍸 {cocktail['name']}")
for ingredient, proportion in cocktail['proportions'].items():
    print(f"  • {proportion} {ingredient}")
```

## 🎛️ Interface Features

### 1. Classic Generation: MULTIPLE
- **Random Creativity Range**: Set minimum and maximum creativity bounds
- **Multiple Cocktails**: Generate 3-10 cocktails at once
- **Varied Results**: Each cocktail uses a random creativity within your range
- **Perfect for**: Exploring different styles and getting inspiration

### 2. Personalized with Ingredients
- **Ingredient Selection**: Choose from categorized ingredient lists
- **Available Ingredients Only**: Creates cocktails using only selected ingredients
- **Smart Combinations**: GNN embeddings ensure compatible ingredient pairing
- **Perfect for**: Using what you have at home

## 🧠 Technical Deep Dive

### GNN Embedding Selection Process

The system uses Graph Neural Network embeddings to make intelligent ingredient choices:

1. **Embedding Computation**: Each ingredient has a 128-dimensional embedding learned from real cocktail data
2. **Similarity Calculation**: Uses cosine similarity between embeddings to find compatible ingredients
3. **Real Data Scoring**: Weights selections based on actual cocktail combinations in the training data
4. **Creativity Temperature**: Higher creativity values increase the influence of embedding similarity vs. proven combinations
5. **Balanced Selection**: Ensures mixology rules while allowing for creative exploration

### Creativity as Temperature
The creativity parameter balances between safe, proven combinations and experimental ingredient selections:

```python
final_score = (
    (1-creativity) * real_combination_score +  # Traditional combinations
    creativity * embedding_similarity +        # GNN-learned compatibility
    balance_bonus                              # Mixology rules
)
```

### Mixology Rules Enforcement
- **Max 1 spirit** per cocktail (gin, vodka, rum, etc.)
- **Max 1 liqueur** for balance
- **Realistic proportions** based on ingredient type:
  - Spirits: 32-45ml
  - Liqueurs: 15-30ml
  - Citrus: 10-20ml
  - Syrups: 5-15ml
  - Bitters: 2-4 dashes

## 📊 Model Performance

- **Node Embeddings**: 128-dimensional learned representations
- **Graph Coverage**: 324 real cocktail-ingredient relationships
- **Ingredient Similarity**: Cosine similarity in embedding space
- **Training Data**: Authentic cocktail recipes from TheCocktailDB

### Example Generated Cocktails
```
🍸 The Crimson Symphony
• **52ml** gin
• **18ml** elderflower cordial
• **12ml** lime juice
• **3 dashes** orange bitters
• **1 twist** orange peel

🍸 Midnight Vodka Dream
• **48ml** vodka
• **22ml** cointreau
• **15ml** cranberry juice
• **1** cherry
```

## 🛠️ Technical Stack

- **PyTorch Geometric**: Graph neural network implementation
- **NetworkX**: Graph manipulation and analysis
- **Pandas**: Data processing and analysis
- **NumPy**: Numerical computations
- **Gradio**: Web interface
- **Matplotlib**: Data visualization

## 📁 Project Structure

```
cocktail_gnn/
├── data/
│   ├── processed/           # Processed datasets & trained models
│   └── raw/                 # Raw cocktail data
├── src/
│   ├── data/
│   │   ├── collectors.py    # Data collection from APIs
│   │   └── graph_builder.py # Graph construction
│   ├── models/
│   │   ├── cocktails_gnn.py # GNN model training
│   │   └── cocktails_generator.py # Cocktail generation
│   └── interface_cocktails.py # Gradio web interface
├── notebooks/               # Jupyter notebooks for exploration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔬 Key Algorithms

### GraphSAGE Node Embeddings
The system uses GraphSAGE (Graph Sample and Aggregate) to learn ingredient embeddings:
- **3-layer** message passing architecture
- **128-dimensional** ingredient embeddings
- **Learned from real cocktail data** to capture ingredient relationships

### Ingredient Selection Algorithm
1. **Validate mixology rules** (alcohol limits)
2. **Calculate embedding similarity** with current ingredients
3. **Score real combinations** from training data
4. **Apply complement bonuses** for balance
5. **Select best candidate** with weighted scoring using creativity as temperature

### Proportion Generation
- **Type-based rules**: Different measures for spirits, liqueurs, citrus
- **Real data analysis**: Based on actual cocktail proportions
- **Randomization**: Natural variation within realistic ranges

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Leonard Havet - leonard.havet@gmail.com

