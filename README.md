# Introduction

We've all done K-Means clustering examples and the results look good, but are they right, and what are other alternative Unsupervised Clustering Techniques

# Data
Kaggle, as we all know, is a wealth of data sets, do got this project I will use a Survey dataset from there.
Specifically 'Mental Health in Tech Survey' - https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey
I have included a copy of the data in this repo for convenience.

# Comparisons performed
I performed comparisons of outputs with multiple approaches.
Namely:
- K-Means (via one-hot encoding and scaling), PCA and Random Forest to identify Feature Importance
- K-Means (via LLM embeddings and scaling), PCA and Random Forest to identify Feature Importance
- K-Mode (via one-hot encoding) and MCA and Random Forest to identify Feature Importance
- Latent Class Analysis (LCA)

# Notebooks
In my comparisons, I used a number of notebooks to test out the approaches above.

# Run the app in streamlit
For convenience I put everything into a simple streamlit app
It can be run locally: `streamlit run streamlit_clustering.py`
Or at this HuggingFace Space () or Streamlit Cloud ().
