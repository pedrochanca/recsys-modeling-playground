## Metrics

This module implements standard evaluation metrics for Recommender Systems, focusing on 
both **Target Prediction** (Accuracy) and **Ranking Quality** (Relevance).

---

### Binary Relevance

Most ranking metrics (Precision, Recall, NDCG, Hit Rate) require classifying items as 
"Relevant" or "Not Relevant."

Here a **Thresholding** approach is used:
* **Relevant (1)**: True Rating $\ge$ `threshold`
* **Not Relevant (0)**: True Rating $<$ `threshold`

The threshold depends on the target variable being considered. For example:
* Rating (**explicit feedback**): values vary between 0 and 5 $\rightarrow$ threshold = 3.5
* Click (**implicit feedback**): binary value 0 / 1 $\rightarrow$ threshold = 0.5

#### Top-K Evaluation

Ranking metrics are computed on the **Top-K** items predicted by the model for each user.
1.  All test items for a user are sorted by their **Predicted Score** (descending).
2.  The top $K$ items are selected as the recommendation list.
3.  The **True Ratings** of these $K$ items are compared against the threshold.

--- 

### Regression-based

#### RMSE (Root Mean Squared Error)
**Type**: Regression / Accuracy
**Goal**: Measures how close the predicted ratings are to the true ratings.

$$
RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{pred, i} - y_{true, i})^2}
$$

* **Logic**: Calculated globally across **ALL** test samples for all users. It does not depend on ranking or the threshold.
* **Interpretation**: Lower is better. 0.0 is perfect accuracy.

--- 

### Ranking-based

#### Precision@K
**Type**: Ranking
**Goal**: Measures the proportion of relevant items within the recommended Top-K list.

$$
Precision@K = \frac{\text{Number of Relevant Items in Top K}}{K}
$$

* **Logic**:
    * **Numerator**: Count of items in Top-K where $TrueRating \ge Threshold$.
    * **Denominator**: The fixed cutoff $K$. (Note: If a user has fewer than $K$ test items, the denominator remains $K$ to penalize the inability to fill the list).
* **Interpretation**: Higher is better. A score of 1.0 means every recommended item was "Relevant."

---

#### Recall@K
**Type**: Ranking
**Goal**: Measures how many relevant items the model found, compared to how many exist in total for that user.

$$
Recall@K = \frac{\text{Number of Relevant Items in Top K}}{\text{Total Number of Relevant Items}}
$$

* **Logic**:
    * **Numerator**: Count of items in Top-K where $TrueRating \ge Threshold$.
    * **Denominator**: Total count of items in the user's test set where $TrueRating \ge Threshold$.
    * *Edge Case*: If a user has 0 relevant items total, Recall is returned as `None` (or 0.0 depending on aggregation) to avoid division by zero.
* **Interpretation**: Higher is better. A score of 1.0 means the model successfully found every single "Relevant" item available.

---

#### Hit Rate@K
**Type**: Ranking
**Goal**: A binary metric indicating if *at least one* relevant item appeared in the recommendation list.

$$
HitRate@K = \mathbb{I}(\text{Number of Relevant Items in Top K} > 0)
$$

* **Logic**: Returns `1.0` if there is at least one item in Top-K with $TrueRating \ge Threshold$, otherwise `0.0`.
* **Interpretation**: Higher is better. Useful for systems where users only need one good suggestion to be satisfied (e.g., "Watch this next").

---

#### NDCG@K (Normalized Discounted Cumulative Gain)
**Type**: Ranking
**Goal**: Measures ranking quality, giving higher importance to relevant items appearing **higher** up the list.

$$
NDCG@K = \frac{DCG@K}{IDCG@K}
$$

**DCG (Discounted Cumulative Gain)**:
$$
DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}
$$
*Where $rel_i$ is 1 if relevant, 0 otherwise. The term $\log_2(i+1)$ (or standard $i+2$ indexing) acts as a penalty discount that grows larger as rank decreases.*

**IDCG (Ideal DCG)**:
The DCG score of a hypothetical "Perfect Ordering" where all relevant items are placed at the very top of the list.

* **Example**:
    * Actual Relevant Item List: [Item A (1), Item B (0), Item C (1)] $\rightarrow$ DCG = 1.5
    * Ideal Relevant Item List: [Item A (1), Item C (1), Item B (0)] $\rightarrow$ DCG = 1.63

* **Logic**:
    * Uses Binary Relevance (0 or 1).
    * *Edge Case*: If a user has 0 relevant items, IDCG is 0, so NDCG is defined as 0.0 (or excluded).
* **Interpretation**: Higher is better. 1.0 is a perfect ranking.

---

### Usage

The main entry point is `compute_metrics`.

```python
results = compute_metrics(
    user_pred_true=user_predictions_dict, 
    metrics=["precision", "recall", "ndcg", "rmse"],  # metrics to run
    k=10,                                             # Top-10 items
    threshold=3.5                                     # Items >= 3.5 are "Relevant"
)

print(results)
# Output: {'precision': 0.45, 'recall': 0.12, 'ndcg': 0.38, 'rmse': 0.89}```