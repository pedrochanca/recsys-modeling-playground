# Graphs For Recommendation - Exploration

![Dates](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pedrochanca/graphs-for-recommendation-experimentation/badges/dates.json)

In this repo, you will find different experiments on Graph-based models for recommendation systems. 

The goal is to have a better understanding of how these architectures work, starting with standard baselines and moving toward Graph Neural Networks (GNNs).

--------------------------------------------------------------------


## Prerequisites
* **Python 3.12+**

--------------------------------------------------------------------


## Setup 

Install requirements (e.g., inside a `pyenv` or `conda` environment):

```bash
pip install -r requirements.txt
```

--------------------------------------------------------------------


## Example to run:
```
python -m scripts.run_ncf --model DeepNCF --tune --plot --verbose
```

--------------------------------------------------------------------

## Libraries

- PyTorch
- Pandas
- Scikit-learn (for Label Encoding)
- Matplotlib (visualization)

--------------------------------------------------------------------

## Dataset

1. MovieLens Latest Small dataset
    - Used for an Explicit Feedback w/ NCF.
    - 100k ratings
    - 610 unique users
    - 9,724 unique items (movies)

2. Amazon Reviews (books)
    - To be used for Implicit Feedback.
    - Link: https://amazon-reviews-2023.github.io

--------------------------------------------------------------------

## Models

So far we have implemented two versions of the **Neural Collaborative Filtering (NCF)** [1] approach. 
- **SimpleNCF**: simplied version of the NCF using a single-layer Multi-Layer Perceptron (MLP).
- **DeepNCF**: a variant of the MLP framework proposed in the NCF paper [1], where an MLP "tower" structure is adopted.

--------------------------------------------------------------------

## Next Steps

- NCF (MLP version) beyond embedding look-ups (features)

- NCF: Explore other datasets, including those w/ implicit feedback

--------------------------------------------------------------------


## References

[1] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, "Neural collaborative filtering," in Proceedings of the 26th International Conference on World Wide Web, 2017, pp. 173–182. doi: 10.1145/3038912.3052569.