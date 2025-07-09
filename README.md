# 🍸 AI Cocktail Generator with Graph Neural Networks

A sophisticated cocktail recipe generator powered by Graph Neural Networks (GNN) that learns from real cocktail data to create new, balanced, and realistic cocktail recipes.

**🚀 [Try the Live Demo on Hugging Face](https://huggingface.co/spaces/Paataatee/Cocktails_GNN)**

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Graph Neural Network**: Uses GraphSAGE to learn cocktail-ingredient relationships
- **Real Data Training**: Trained on 97 authentic cocktail recipes from TheCocktailDB
- **Mixology Rules**: Enforces realistic cocktail composition rules (max 1 spirit + 1 liqueur)
- **Smart Proportions**: Generates realistic measurements based on cocktail type
- **Interactive Interface**: Gradio web interface for easy cocktail generation
- **Creativity Control**: Adjustable creativity levels from classic to experimental
- **Novelty Scoring**: Evaluates how unique generated cocktails are

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
- **Link prediction** training objective

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

#### 1. Data Collection & Processing
```bash
# Collect cocktail data from TheCocktailDB
python src/data/collectors.py

# Build the cocktail-ingredient graph
python src/data/graph_builder.py
```

#### 2. Train the GNN Model
```bash
# Train GraphSAGE on cocktail data
python src/models/cocktails_gnn.py
```

#### 3. Generate Cocktails
```bash
# Launch interactive interface
python src/interface_cocktails.py

# Or generate programmatically
python src/models/cocktails_generator.py
```

#### 4. Explore Data (Optional)
```bash
# Open Jupyter notebook for data exploration
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 💡 How It Works

### 1. Data Collection
- Fetches cocktail recipes from TheCocktailDB API
- Filters for alcoholic cocktails only
- Normalizes ingredient names and measures

### 2. Graph Construction
- Creates bipartite graph: cocktails ↔ ingredients
- Stores real proportions as edge weights
- Analyzes ingredient frequencies and patterns

### 3. GNN Training
- Learns embeddings for all nodes (cocktails + ingredients)
- Predicts which ingredients work well together
- Uses link prediction with negative sampling

### 4. Cocktail Generation
```python
from models.cocktails_generator import CocktailGenerator

generator = CocktailGenerator()

# Generate a gin-based cocktail
cocktail = generator.generate_cocktail_from_base(
    base_ingredient="gin",
    creativity=0.7,
    num_ingredients=4
)

print(f"🍸 {cocktail['name']}")
print("Ingredients:")
for ingredient, proportion in cocktail['proportions'].items():
    print(f"  • {proportion} {ingredient}")
```

### 5. Mixology Rules Enforcement
- **Max 1 spirit** per cocktail (gin, vodka, rum, etc.)
- **Max 1 liqueur** for balance
- **Realistic proportions** based on cocktail type:
  - Spirits: 45-60ml
  - Liqueurs: 15-30ml
  - Citrus: 10-20ml
  - Syrups: 5-15ml
  - Bitters: 2-4 dashes

## 🎛️ Interface

The Gradio interface provides two modes:

### Single Cocktail Generation
- Choose spirit base (gin, vodka, rum, etc.)
- Adjust creativity level (0 = classic, 1 = experimental)
- Set number of ingredients
- Generate balanced cocktail with instructions

### Multiple Cocktail Generation
- Generate 5 cocktails at once
- Random or specific spirit selection
- Varying creativity levels
- Perfect for cocktail menu creation

## 📊 Model Performance

- **Node Embeddings**: 128-dimensional learned representations
- **Graph Coverage**: 324 real cocktail-ingredient relationships
- **Ingredient Similarity**: Cosine similarity in embedding space
- **Novelty Scores**: Jaccard similarity with existing cocktails

### Example Generated Cocktails
```
🍸 The Crimson Symphony
• 52ml gin
• 18ml elderflower cordial
• 12ml lime juice
• 3 dashes orange bitters
• 1 twist orange peel

🍸 Midnight Vodka Dream
• 48ml vodka
• 22ml cointreau
• 15ml cranberry juice
• 1 cherry
```

## 🛠️ Technical Stack

- **PyTorch Geometric**: Graph neural network implementation
- **NetworkX**: Graph manipulation and analysis
- **Pandas**: Data processing and analysis
- **NumPy**: Numerical computations
- **Gradio**: Web interface
- **Requests**: API data collection
- **Matplotlib**: Data visualization

## 📁 Project Structure

```
cocktail_gnn/
├── data/
│   ├── raw/                 # Raw cocktail data
│   ├── processed/           # Processed datasets & models
│   └── external/            # External data sources
├── src/
│   ├── data/
│   │   ├── collectors.py    # Data collection from APIs
│   │   └── graph_builder.py # Graph construction
│   ├── models/
│   │   ├── cocktails_gnn.py # GNN model training
│   │   └── cocktails_generator.py # Cocktail generation
│   ├── evaluation/          # Model evaluation
│   └── utils/               # Utility functions
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔬 Key Algorithms

### GraphSAGE Node Embeddings
```python
class CocktailGNN(nn.Module):
    def __init__(self, num_nodes, input_dim, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, input_dim)
        self.convs = nn.ModuleList([
            SAGEConv(input_dim if i == 0 else hidden_dim, 
                    hidden_dim if i < num_layers-1 else embedding_dim)
            for i in range(num_layers)
        ])
```

### Ingredient Selection Algorithm
1. **Validate mixology rules** (alcohol limits)
2. **Calculate embedding similarity** with current ingredients
3. **Score real combinations** from training data
4. **Apply complement bonuses** for balance
5. **Select best candidate** with weighted scoring

### Proportion Generation
- **Type-based rules**: Different measures for spirits, liqueurs, citrus
- **Real data analysis**: Based on actual cocktail proportions
- **Randomization**: Natural variation within realistic ranges

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Contact

Leonard Havet - leonard.havet@gmail.com

