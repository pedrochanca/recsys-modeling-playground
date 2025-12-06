# Graphs For Recommendation - Exploration

**Created At:**  || **Last Updated:** 

In this repo, you will find different experiments on Graph-based models for recommendation systems. 

The goal is to have a better understanding of how these architectures work, starting with standard baselines and moving toward Graph Neural Networks (GNNs).

## Prerequisites
* **Python 3.12+**

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

Here we implement a simplified version of the **Neural Collaborative Filtering (NCF)** [1] approach using a single-layer Multi-Layer Perceptron (MLP).

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

--------------------------------------------------------------------

## DeepNCF (Multi-Layer Perceptron)

To capture complex, non-linear interactions between user and item latent vectors, this model employs a Multi-Layer Perceptron (MLP) "tower" structure instead of the single linear layer used in the baseline model. This architecture directly adopts the MLP framework proposed in Neural Collaborative Filtering paper [1].

For evaluation, we utilize the specific configuration of hidden layer dimensions that yielded the optimal performance during hyperparameter tuning.

![DeepNCF Architecture diagram](images/deep_ncf_architecture.jpg)

### Model Architecture

The architecture extends the baseline by adding hidden layers with ReLU activations.

#### The Forward Pass

1.  **Input**: User ID and Item ID.
2.  **Embedding Lookup**: IDs are mapped to dense vectors (size 32).
3.  **Concatenation**: User and Item vectors are joined (size 64).
4.  **MLP Layers**:
    * **Layer 1 (Input)**: Linear ($64 \to 32$) + ReLU + Dropout
    * **Layer 2 (Hidden)**: Linear ($32 \to 16$) + ReLU + Dropout
    * **Layer 3 (Output)**: Linear ($16 \to 1$)

#### Mathematical Formulation

For user $u$ and item $i$, the input vector $\mathbf{x}_0$ is the concatenated embedding. The deep network transforms this input through $L$ layers. For a hidden layer $l$:

$$
\mathbf{x}_l = \sigma(\mathbf{W}_l \mathbf{x}_{l-1} + \mathbf{b}_l)
$$

Where $\sigma$ is the activation function (ReLU). The final score $\hat{y}_{u,i}$ is produced by the output layer:

$$
\hat{y}_{u,i} = \mathbf{W}_{out} \mathbf{x}_{last} + b_{out}
$$


--------------------------------------------------------------------


## Next Steps

- NCF (MLP version)

- NCF (MLP version) beyond embedding look-ups (features)
    - Potentially try out other datasets

- Metrics used (which and why)

## References

[1] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, "Neural collaborative filtering," in Proceedings of the 26th International Conference on World Wide Web, 2017, pp. 173–182. doi: 10.1145/3038912.3052569.