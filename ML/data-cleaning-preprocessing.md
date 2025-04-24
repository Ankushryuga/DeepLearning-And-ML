# Data cleaning: 
its a important step of ML pipeline as it involves identifying and removing any missing duplicate or irrelevant data.
The goal of data cleaning is to ensure that the data is accurate, consistent and free of errors as raw data is often noisy,
incomplete and incosistent which can -vely impact the accuracy of model and its reliability of insights derived from it.

## How to perform data cleanliness?
![image](https://github.com/user-attachments/assets/887730c7-78cc-4039-9274-f2fbf58ef5bd)


## Example
import pandas as pd
import numpy as np
nfl_data=pd.read_csv("../nfl_data.csv")
np.random.seed(0)  

-> take a look at new dataset.
nfl_data.head()

-> How many missing data points do we have?
missing_values_count=nfl_data.isnull().sum()
missing_values_count[0:10]
o/p:
Date                0
GameID              0
Drive               0
qtr                 0
down            61154
time              224
TimeUnder           0
TimeSecs          224
PlayTimeDiff      444
SideofField       528
dtype: int64
--> percent:
total_cell=np.product(nfl_data.shape)
total_missing=missing_values_count.sum()

--> percent of missing :
percent_missing=(total_missing/total_cells)*100
print(percent_missing)


# Figure out why the data is missing?
1. is this value missing because it wasn't recorded or because it doesn't exist?

**If a value is missing becuase it doesn't exist (like the height of the oldest child of someone who doesn't have any children) then it doesn't make sense to try and guess what it might be. These values you probably do want to keep as NaN. On the other hand, if a value is missing because it wasn't recorded, then you can try to guess what it might have been based on the other values in that column and row. This is called ::::imputation**



### Handling Missing Data:::
## Loading data..
import pandas as pd
import numpy as np

nfl_data=pd.read_csv("NFL Play by Play 2009-2017 (v4).csv") 
np.random.seed(0)
nfl_data.head()

## Finding the missing values..
missing_values_count=nfl_data.isnull().sum()
missing_values_count[0:10]
total_cells = np.prod(nfl_data.shape)
# print(total_cells)
total_missing = missing_values_count.sum()
# print(total_missing)
percent_missing=(total_missing/total_cells)*100
print(percent_missing)
## First apporach of handling missing values:: 1. By dropping...
## Droping the missing values( Not recommended )..
columns_with_na_dropped=nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
columns_with_na_dropped.shape
print(columns_with_na_dropped)
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d " % columns_with_na_dropped.shape[1])

# print(nfl_data.shape)
# print(columns_with_na_dropped.shape)
NOTE: axis=1 means column, and axis=0 means row.

## 2nd Approach: By filling the missing values automatically..
subset_nfl_data=nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
subset_nfl_data.fillna(0) # replace all NA's with 0.
subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)  ## replace all NA's the value that comes directly after it in the same column.


##   Scaling and Normalization
Import neccessary libraries:

#modules
import pandas as pd
import numpy as np
#for Box-Cox transformations
from scipy import stats
#for min_max scaling
from mlxtend.preprocessing import minmax_scaling
#plotting modules
import seaborn as sns
import matplotlib.pyplot as plt
#set seed for reproducibility.
np.random.seed(0)


## Scaling vs Normalization: what's the difference?
-> Scaling:- you're changing the range of your data
-> Normalization:- you're changing the shape of the distribution of your data

# Scaling:
it means that you're transforming your data so that it fits within a specific scale, like 0-100 or 0-1. you want to scale data when you're using methods based on measures of how far apart data points are, like **support vector machines(SVM)** or **k-nearest neighbors**. with these algorithms, a change of "1" in any numeric feature is given the same importance.

==>For example, you might be looking at the prices of some products in both Yen and US Dollars. One US Dollar is worth about 100 Yen, but if you don't scale your prices, methods like SVM or KNN will consider a difference in price of 1 Yen as important as a difference of 1 US Dollar! This clearly doesn't fit with our intuitions of the world. With currency, you can convert between currencies. But what about if you're looking at something like height and weight? It's not entirely clear how many pounds should equal one inch (or how many kilograms should equal one meter).

By scaling your variables, you can help compare different variables on equal footing. 


# generate 1000 data points randomly drawn from an exponential distribution..
original_data=np.random.exponential(size=1000)
# mix-max scale the data b/w 0 and 1.
scaled_data=minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax=plt.subplots(1,2, figsize=(15,3))
sns.hisplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled Data")
plt.show()

Notice: the shape of the data doesn't change, but that instead of ranging from 0 to 8ish, it now ranges from 0 to 1.

## Normalization:
Scaling just changes the range of your data. Normalization is a more radical transformation. The point of normalization is to change your observations so that they can be described as a normal distribution.

=> Normal Distribution: Also known as the "bell curve", this is a specific statistical distribution where a roughly equal observations fall above and below the mean, the mean and the median are the same, and there are more observations closer to the mean. The normal distribution is also known as the Gaussian distribution.

in general, you'll normalize your data if you're going to be using a ML or statistics techinque that assumes your data is normally distributed. some exmple of these include linear discriminant analysis (LDA) and Gaussian naive Bayes (any method with "Gaussian" in the name probably  assumes normality).



## BOX-COX Transformation
The one-parameter Box–Cox transformations are defined as

![image](https://github.com/user-attachments/assets/2e5ed5c0-5340-482a-88f2-21022c7a82ec)

and the two-parameter Box–Cox transformations as
![image](https://github.com/user-attachments/assets/102bec27-5055-46dd-abc5-bdd3263252f1)

The parameter ![image](https://github.com/user-attachments/assets/d469dc20-2a80-4a9c-bfd7-eb4ddc5e1003) is estimated using the profile likelihood function and using goodness-of-fit tests.

# Normalize the exponential data with boxcox
normalized_data=stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2, figsize=(15,3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")
plt.show()

Notice that the shape of our data has changed. Before normalizing it was almost L-shaped. But after normalizing it looks more like the outline of a bell (hence "bell curve").




