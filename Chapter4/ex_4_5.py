''' 
Q: Consider you have applied meta-labels to events determined by a trend-following model. Suppose that two thirds of the labels are 0 and one third 
of the labels are 1.

(a) What happens if you fit a classifier without balancing class weights?

- The classifier exploits class imbalance to maximize accuracy by defaulting heavily toward
the majority class. Since 2/3rds of the labels are 0, the model can achieve 67% accuracy by
predicting 0 almost everywhere. Therefore the loss function rewards a decision boundary that 
demands strong evidence before predicting 1. The model will have high accuracy but low recall
on class 1 since it misses most true positives. The predicted probabilities for class 1 are 
around 0.33 while the meta model recommends skipping every signal (defeats the purpose of 
having the two models).

(b) A label 1 means true positive, and a label 0 means false positive. By applying balanced 
class weights we force the classifier to pay more attention to true positives and less 
attention to false positives. Why does that make sense?

- the cost structure of meta-labeling is asymmetric in a way that the default loss function 
doesn't see. A false negative carries real opportunity cost, while a true negative would have
been skipped anyway. The goal of meta-labeling should be to identify rare winners, not confirm 
losers are losers. Also by upweighting class 1 the classifier can value identifying a true 
positive more than a false positive. The primary objective is to identify the positives and 
filtering out the negative signals is just a byproduct of this.

(c) What is the distribution of predicted labels, before and after applying balanced class weights?

- Before balancing predicted labels are skewed toward 0, with predicted probabilities of class 1 
around the base rate of 0.33, with few observations crossing the 0.5 threshold. After balancing the 
weights, predicted labels shift toward the true class proportions around 55% zerps and 44% ones, 
with predicted probabilities centered near 0.5 with a spread in both directions. The post-balanced
distribution gives the meta-model the dynamic range for sizing: probabilities now meaningfully
determine "definitely skip", "uncertain", etc. On the contrary, the pre-balanced output collapsed
almost everything into the "skip" region.



'''