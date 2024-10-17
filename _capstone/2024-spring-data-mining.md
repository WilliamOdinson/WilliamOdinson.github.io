---
title: "Precision Rating Prediction Restaurant Recommendation System Optimized with Multi-Model Integration"
collection: capstone
type: "Data Warehouse & Data Mining"
permalink: /capstone/2024-spring-data-mining
venue: "Tianjin University"
date: 2024-06-18
location: "Tianjin, China"
excerpt: "Created a precision recommendation tool by integrating NLP and multiple algorithms to eliminate review bias, enhancing accuracy and personalization using the Yelp dataset."
---

This research project aims to develop a precise restaurant recommendation system that eliminates subjective biases in user reviews through multi-model integration, thereby enhancing the accuracy and personalization of recommendations. Our data source is the Yelp dataset, which includes a vast number of user reviews and restaurant information, such as user ratings and review texts.

To effectively handle subjectivity in user reviews, we first employed natural language processing (NLP) techniques. We established a baseline using a Naive Bayes model and then utilized GloVe word embeddings and the Transformer-based DistilBERT model to extract and analyze deep semantic information from review texts. This allowed us to construct models capable of extracting objective rating information from reviews. By combining user ratings with the analysis results from the DistilBERT model and applying a weighted method to optimize the rating data, we aimed to achieve higher accuracy and objectivity. In constructing the recommendation system, we integrated multiple algorithms, including Singular Value Decomposition (SVD), Cosine Similarity, Alternating Least Squares (ALS), Stochastic Gradient Descent (SGD), and Random Forest Regressors. This multi-model integration approach enabled us to leverage the strengths of each model, further combining their prediction results using linear regression and high-penalty Ridge Regression to optimize the final recommendation accuracy. We evaluated the models primarily using Mean Squared Error (MSE), and the results indicated that our integrated model has significant advantages in handling sparse data, effectively reducing prediction errors and enhancing interpretability.