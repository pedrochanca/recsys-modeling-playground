## Neural Collaborative Filtering (NCF) - Code Implementation

Other people have done it -> https://github.com/guoyang9/NCF

#### MLP approach of NCF

##### Here is the step list:

1. **Start with user ID and item ID**
    - Those are your input features (in the paper: only IDs).

2. **One-hot encode user ID and item ID**
    - User → one-hot user vector
    - Item → one-hot item vector

3. **Look up / compute user and item embeddings**
    - Multiply one-hot vectors by embedding matrices → user embedding p_u, item embedding q_i.
    - These embedding matrices are learned parameters.

4. **Concatenate the embeddings**
    - Form a single vector: [p_u; q_i].

5. **Pass this vector through several hidden (MLP) layers**
    - Fully connected layers with nonlinear activations (e.g. ReLU), ending in a final hidden representation.

6. **Output a predicted score**
    - Final layer (with sigmoid) outputs  y^_ui ∈ (0,1) = predicted probability that user u interacts with item i.

7. **Train with labels**
    - Compare y^_ui to the true label y_ui (1 for observed, 0 for sampled negative) using binary cross-entropy.
    - Backpropagate the loss to update both embeddings and MLP weights.

