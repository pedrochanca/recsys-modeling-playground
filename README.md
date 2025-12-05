# Graphs For Recommendation Experimentation

In this repo, you will find different experiments on Graph-based models for recommendation systems. 

The goal is to have a better understanding of how these architectures work, starting with standard baselines and moving toward Graph Neural Networks (GNNs).

## Setup

Install requirements (e.g., inside a `pyenv` or `conda` environment):

```bash
pip install -r requirements.txt
```

## Libraries

- PyTorch
- Pandas
- Scikit-learn (for Label Encoding)
- Matplotlib (visualization)

## Dataset

I will be using the MovieLens Latest Small dataset.
- 100k ratings
- 610 unique users
- 9,724 unique items (movies)

## SimpleNCF

Here we implement a simpler version of the **Neural Collaborative Filtering (NCF)** approach using a single-layer Multi-Layer Perceptron (MLP).

It takes the user and item embeddings, concatenates them, and passes them through a single linear layer to predict the rating.

Instead of using sparse One-Hot Encoded vectors, the model utilizes **Embedding Layers** to map User IDs and Item IDs into dense latent vectors. 

### Model Architecture

#### The Forward Pass

1.  **Input**: User ID and Item ID.
2.  **Embedding Lookup**: IDs are mapped to dense vectors of size `emb_dim=32`.
3.  **Concatenation**: User and Item vectors are joined to form a single input vector of size 64.
4.  **Linear Transformation**: A fully connected layer (`nn.Linear`) applies weights and a bias to output the final scalar prediction.

![SimpleNCF Architecture diagram](images/simple_ncf_architecture.jpg)

#### Mathematical Formulation

For a specific user $u$ and item $i$, we define the embeddings as $\mathbf{e}_u$ and $\mathbf{e}_i$ (both size 32). The input to the model is the concatenation of these vectors:

$$
\mathbf{x}_{u,i} = [\mathbf{e}_u, \mathbf{e}_i] \in \mathbb{R}^{64}
$$

The `nn.Linear` layer applies a learnable weight matrix $\mathbf{W}$ (size $1 \times 64$) and a bias $b$ to produce the predicted rating $\hat{y}_{u,i}$:

$$\hat{y}_{u,i} = \mathbf{W} \mathbf{x}_{u,i} + b$$

This can be expanded as the weighted sum of features:

$$\hat{y}_{u,i} = \sum_{k=1}^{64} (w_k \cdot x_{u,i}^{(k)}) + b$$

#### Notation Legend

* $N_{users}, N_{items}$: Total number of unique users (610) and items (9724).
* $\mathbf{e}_u$: Embedding vector for user $u$ (dimension $1 \times 32$).
* $\mathbf{e}_i$: Embedding vector for item $i$ (dimension $1 \times 32$).
* $\mathbf{x}_{u,i}$: Concatenated feature vector (dimension $1 \times 64$).
* $\mathbf{W}$: Learnable weight matrix of the linear layer (dimension $1 \times 64$).
* $b$: Learnable bias term (scalar).

#### Data Flow Example

Below is a trace of the data shapes and values for a single prediction:

    1. Input: User ID [42], Item ID [101]

    2. Embedding Lookup:
        - User 42 → [0.1, -0.5, ...] (Size 32)
        - Item 101 → [0.9, 0.2, ...] (Size 32)

    3. Concatenation:
        - Combines into one long vector → [0.1, -0.5, ..., 0.9, 0.2, ...] (Size 64)

    4. nn.Linear (The 1-Layer MLP):
        - Calculates weighted sum of those 64 numbers.

    5. Output:
        - Predicted Rating (e.g., 3.5).