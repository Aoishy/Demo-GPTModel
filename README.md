# Demo-GPTModel

A PyTorch implementation of a GPT (Generative Pre-trained Transformer) model built from scratch. This project demonstrates the core components of transformer architecture including multi-head self-attention, feed-forward networks, and text generation capabilities.

## 🌟 Features

- **Complete GPT Architecture**: Implementation of transformer blocks with multi-head self-attention
- **Custom Dataset Handling**: Efficient data loading and tokenization using tiktoken
- **Text Generation**: Simple text generation with temperature control
- **Pre-training Pipeline**: Training loop with loss calculation and optimization
- **Modular Design**: Clean separation of model components for easy understanding and modification

## 🏗️ Architecture

The model consists of the following key components:

### Core Components
- **MultiHeadAttention**: Self-attention mechanism with multiple attention heads
- **FeedForward**: Position-wise feed-forward network with GELU activation
- **TransformerBlock**: Complete transformer block with layer normalization and residual connections
- **GPTModel**: Main model class combining all components

### Model Configuration
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-key-value bias
}
```

## 📁 Project Structure

```
Demo-GPTModel/
├── gptModel.py          # Core GPT model implementation
├── loadData.py          # Data loading and preprocessing utilities
├── preTrainfirst.py     # Pre-training pipeline and training utilities
├── testPrompt.py        # Text generation and model testing
├── the-verdict.txt      # Sample training text data
├── model_and_optimizer.pth  # Saved model checkpoint
└── README.md           # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- tiktoken
- urllib (for downloading sample data)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Aoishy/Demo-GPTModel.git
cd Demo-GPTModel
```

2. Install required packages:
```bash
pip install torch tiktoken
```

### Usage

#### 1. Data Preparation
The model uses "The Verdict" text for training. The data is automatically downloaded if not present:

```python
from loadData import create_dataloader_v1

# Create data loader
dataloader = create_dataloader_v1(
    txt=text_data, 
    batch_size=4, 
    max_length=256,
    stride=128
)
```

#### 2. Model Training
```python
from preTrainfirst import train_model_simple

# Initialize model
model = GPTModel(GPT_CONFIG_124M)

# Train the model
train_model_simple(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    device, 
    num_epochs=10
)
```

#### 3. Text Generation
```python
from testPrompt import generate_text_simple
import tiktoken

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Generate text
start_context = "Every effort moves you"
context_size = model.pos_emb.num_embeddings
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=50,
    context_size=context_size
)

print(token_ids_to_text(token_ids, tokenizer))
```

## 🔧 Model Components

### Multi-Head Attention
Implements scaled dot-product attention with multiple heads and causal masking for autoregressive generation.

### Feed-Forward Network
Two-layer MLP with GELU activation function, expanding the hidden dimension by 4x.

### Layer Normalization
Custom implementation of layer normalization for training stability.

### GELU Activation
Gaussian Error Linear Unit activation function for improved model performance.

## 📊 Training

The training pipeline includes:
- **Loss Calculation**: Cross-entropy loss for next-token prediction
- **Optimization**: Adam optimizer with learning rate scheduling
- **Validation**: Separate validation set for monitoring overfitting
- **Checkpointing**: Model and optimizer state saving

## 🎯 Key Features

- **Causal Masking**: Ensures autoregressive property during training
- **Positional Encoding**: Learnable positional embeddings
- **Dropout**: Regularization to prevent overfitting
- **Residual Connections**: Skip connections for better gradient flow
- **Layer Normalization**: Stable training with normalized activations

## 📈 Performance

The model is configured similar to GPT-2 small (124M parameters) and can be scaled up by modifying the configuration parameters.

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the model architecture
- Adding new features
- Optimizing training performance
- Adding more comprehensive documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Inspired by the original GPT paper and OpenAI's implementation
- Built following transformer architecture principles
- Uses tiktoken for efficient tokenization

## 📚 References

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) for training data

---

⭐ **Star this repository if you find it helpful!**

