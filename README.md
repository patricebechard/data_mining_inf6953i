# INF6953I - Data Mining - Summer 2018

Various assignments and exercises I did for the Data Mining class at Polytechnique Montreal during the Summer 2018 semester.

### Laboratoire 1

Assignment on data preprocessing. Used for sentiment analysis on the IMDB Review Dataset. The different preprocessing used on the text data were :

* Bag of Words
* Doc2Vec embeddings

We then trained a logistic regression model as well as a simple MLP neural net on the preprocessed data.

### Laboratoire 2

Assignment on bagging and boosting. Used on the Porto Seguro Save Driver Prediction Dataset. We used various simple algorithms to predict the probability that a driver will initiate an auto insurance claim in the next year. We used the Gini coefficient to evaluate our model.

We used various boosting algorithms such as `AdaBoost` and `eXtreme Gradient Boosting`. We also used bagging techniques to enhance our models.


### Laboratoire 3

Assignment on distributed computing and the MapReduce framework. We used PySpark and the Instacart dataset to analyze the shopping habits of consumers. We implemented a Market Basket Analysis algorithm to do so.

### Outliers

Implementation of a K-means model for outlier detection in a toy dataset.

### PageRank

Implementation of the PageRank algorithm. Tested a small toy network.