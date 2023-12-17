
# FINAL EXAM 

# part a and b
#dataset_final.csv is the dataset containing answers to a personality questionaire of 21644 individuals 

# part c
python3 run_part_c.py
#execute run_part_c.py which performs factor analysis on dataset_final.csv and exports the results into c_results.csv

# part d
python3 run_part_d.py
#execute run_part_d.py which clusters the individuals using k mean clustering and kmedoids to discover the optimal number of clusters and the best technique to cluster the indivduals. run_part_d.py also generates a pairplot and a scatterplot of the optimal number of clusters

# part e
#calculating the inertia value with the kmc model shows that the optimal number of clusters for the dataset is 2. high silhouette scores show that an object is very similar to its own cluster compared to others. the kmc model reported a higher silhouette score than the kmedoids model as an alternative. this shows that the kmc model is a better fit of the data

# part f
python3 run_part_f.py
#execute run_part_f.py which performs linear regression and regeresses math ability on the personality traits generated in part c (c_results.csv). this code also utilizes the kfold template to determine how well the model fits the data (r2).

# part g
#a linear regression model fits better here due to the continuous variables in the model. i first checked with a heat map whether the factors were correlated with math ability. it appears there is not a significant correlation between the independent and dependant variables. i ran a liner regression anyway and trained the machine with the personality traits dataset and the math ability from the original dataset. i then tested the machine against the same dataset to predict math ability. the reported r2's are very low suggesting that the model does not explain much variation in the data, hence there is not signifcant correlation.

# part h
#from the heat map results in part f, it appears that, while not significantly correlated, the third factor (personality trait) is most highly correlated with math ability. going back to the factor loadings from part c, the way to understand which 20 questions to use to maximize the math ability of the people you choose is to see which questions have the highest factor loadings for the personality trait that is most highly associated with math ability, in this case the third trait. you would then need to collect the answers to the questions and run factor analysis on the results to get the 30 individuals who are determined to have that personality trait

# part i 
#in order to get a variety of different personality traits, this answer would not be the same as the previous answer, due to the fact that you now want a variety of different personality traits instead of targeting the single trait that is most highly associated with math ability. in this case you would view the factor loadings for each trait from part c and choose among those factor loadings a subset of 20 questions with an approximately equal amount of questions that have the highest factor loadings for each of the three personality traits. you would then conduct factor analysis on the results and choose among the individuals a sufficient amount of each personality traits. 