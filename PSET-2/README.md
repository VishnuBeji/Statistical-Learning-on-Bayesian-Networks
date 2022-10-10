## Statistical Learning of Bayesian Networks
We implement Chow_Liu Algorithm on the ALARM medical dataset to obtain a Bayesian Network representing relationships between the features. 
Chow_Liu boils down to computing the maximum spanning tree from a graph, with nodes being the features and the edges being the mutual informations. We see that maximizing the Mutual Information gives us the best Bayesian Network. 

For the Mutual information we use 

(i) Plugin Estimator: Mutual Information as per: https://en.wikipedia.org/wiki/Mutual_information

(ii) JVHW: Jiao–Venkat–Han–Weissman mutual information estimator
http://web.stanford.edu/~tsachy/index_jvhw.html

Both Methods give us nearly similar edge weights and the resulting trees are same.

For implementing Maximum Spanning Tree, we can use Prim's algorithm or Kruskal's algorithm. Here we have used prim's algorithm.
