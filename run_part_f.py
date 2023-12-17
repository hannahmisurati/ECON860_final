# import the relevant packages
import pandas
from sklearn import linear_model
import kfold_template
import seaborn as sns
import matplotlib.pyplot as plt
# supress warnings
import warnings 
warnings.filterwarnings('ignore')

# reading the dataset
traits = pandas.read_csv("c_results.csv")
print(traits.head())
yvar = pandas.read_csv("dataset_final.csv")


# target is the dependant variable math ability, the independant variables are the personality traits
target = yvar.iloc[:, 40].values
data = traits.iloc[:, 0:3].values

# Creating a new DataFrame with both columns
target_with_data = pandas.DataFrame({'Trait0': data[:, 0], 'Trait1': data[:, 1], 'Trait2': data[:, 2], 'Math': target, })

# Display the updated DataFrame
print(target_with_data)
# print(target)
# print(data)


# plot heatmap to find out if any factors are correlated with math, turns out they are not highly correlated
sns.heatmap( target_with_data.corr(), cmap = 'YlGnBu', annot = True )
plt.show()

# define the linear model and run it anyway
machine = linear_model.LinearRegression()
machine.fit(data,target)


# define same x variables variables to predict math ability
new_dataset = pandas.read_csv("c_results.csv")
new_dataset = new_dataset.values

#use sklearn to predict y given x's, print prediction
prediction = machine.predict(new_dataset)
print(prediction)
print('\n')


# run with kfold template to get an r2 measure, 
# r2 is low which means the model doesn't explain a lot of variation in the data
return_values = kfold_template.run_kfold(machine, data, target, 4, True)
print(return_values)
