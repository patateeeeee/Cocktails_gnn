# 🍸 AI Cocktail Generator with Graph Neural Networks

**🚀 [Try the Live Demo on Hugging Face](https://huggingface.co/spaces/Paataatee/Cocktails_GNN)**

An AI-powered cocktail recipe generator that uses Graph Neural Networks (GNN) to create new cocktail recipes. The system can generate cocktails either randomly or with specific ingredients you provide, using learned embeddings to intelligently select compatible ingredients.

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
git clone https://github.com/patateeeeee/Cocktails_gnn.git
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



### Example Generated Cocktail
```
🍸 The Crimson Symphony
• **52ml** gin
• **18ml** elderflower cordial
• **12ml** lime juice
• **3 dashes** orange bitters
• **1 twist** orange peel


```

## 🛠️ Technical Stack

- **PyTorch Geometric**: Graph neural network implementation
- **NetworkX**: Graph manipulation and analysis
- **Pandas**: Data processing and analysis
- **NumPy**: Numerical computations
- **Gradio**: Web interface
- **Matplotlib**: Data visualization



## 📧 Contact

Leonard Havet - leonard.havet@gmail.com

