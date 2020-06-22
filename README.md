# Artists Recommendation System: Combining Neural Collaborative Recommendations and Content Based Recommendations

I introduce a Recommendation system that combines:
  1. [Neural Collaborative filtering](https://arxiv.org/abs/1708.05031) that extends the collaborative filtering matrix factorization       technique to a deep Neural Net. 
  2. Content Based recommendations that help understand how users are making decisions.
  
There are four main steps I took to build the final model:
  1. Database: Using EMI Music Kaggle competition \cite{kaggle}, I pre-processed a ratings matrix for the recommendation task.    
  2. Model Design: Combined Neural Pytorch model and content based model to generate top 5 unseen artist to any user. 
  3. Evaluation: loss and accuracy metrics on validation and test sets. 
  4. Interpretation: Understanding what the model is learning by comparing content based recommendations on given artists features versus      Neural model artist embedding matrix. 
  
  ![Neural model structure](https://github.com/EdenBD/artists-recommender/blob/master/neural_model.png)
