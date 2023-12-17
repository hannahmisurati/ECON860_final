# import relevant packages
import pandas
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import numpy
import matplotlib.pyplot as plt 


# define dataset and get a subset of the personality questions
dataset = pandas.read_csv("dataset_final.csv")

traits = dataset.iloc[:,0:40]

print(traits)

#test whether there is correlation in the data with bartlett and p-value
chi2 ,p=calculate_bartlett_sphericity(traits)
print(chi2, p)

#test whether this data is suitable for factor analysis, kmo value greater than .6 
kmo_all,kmo_model=calculate_kmo(traits)
print(kmo_model)

# create factor analysis object and perform factor analysis
machine = FactorAnalyzer(n_factors=25, rotation=None)
machine.fit(traits)
# check eigenvalues, look for values greater than 1
ev, v = machine.get_eigenvalues()
#print(ev)

# use scree plot to find where the eigenvalues level off, finds 4 major factors
plt.scatter(range(1,traits.shape[1]+1),ev)
plt.plot(range(1,traits.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
#plt.show()


# choose n_factors = 4 because there were 4 eigenvalues greater than 1 
machine = FactorAnalyzer(n_factors=4, rotation='varimax')
machine.fit(traits)
print(pandas.DataFrame(machine.loadings_,index=traits.columns))
#output = machine.loadings_
#print(output)

#the fourth factor does not have high factor loadings across any questions, so factors were reduced to three
machine = FactorAnalyzer(n_factors=3, rotation='varimax')
machine.fit(traits)
print(pandas.DataFrame(machine.loadings_,index=traits.columns))
factor_loadings = machine.loadings_
# print(factor_loadings)

traits = traits.values
# print(traits)

#perform dot product with traits and output 
results = numpy.dot(traits, factor_loadings)

# export to csv file
pandas.DataFrame(results).round().to_csv("c_results.csv", index=False)








