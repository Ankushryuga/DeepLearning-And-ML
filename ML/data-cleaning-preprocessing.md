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

## Practice on Scaling & Normalization:
from learntools.core import binder
binder.bind(globals())
from learntools.data_cleaning.ex2 import *
print("Setup Complete")

# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)


# select the usd_goal_real column
original_data = pd.DataFrame(kickstarters_2017.usd_goal_real)

# scale the goals from 0 to 1
scaled_data = minmax_scaling(original_data, columns=['usd_goal_real'])

print('Original data\nPreview:\n', original_data.head())
print('Minimum value:', float(original_data.min()),
      '\nMaximum value:', float(original_data.max()))
print('_'*30)

print('\nScaled data\nPreview:\n', scaled_data.head())
print('Minimum value:', float(scaled_data.min()),
      '\nMaximum value:', float(scaled_data.max()))


# Scale goal column..
original_goal_data = pd.DataFrame(kickstarters_2017.goal)

# TODO: Your code here
scaled_goal_data = minmax_scaling(original_goal_data, columns=['goal'])

# Check your answer
q1.check()


## practice Normalization
# get the index of all positive pledges (Box-Cox only takes positive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = pd.Series(stats.boxcox(positive_pledges)[0], 
                               name='usd_pledged_real', index=positive_pledges.index)

print('Original data\nPreview:\n', positive_pledges.head())
print('Minimum value:', float(positive_pledges.min()),
      '\nMaximum value:', float(positive_pledges.max()))
print('_'*30)

print('\nNormalized data\nPreview:\n', normalized_pledges.head())
print('Minimum value:', float(normalized_pledges.min()),
      '\nMaximum value:', float(normalized_pledges.max()))


# plot normalized data
ax = sns.histplot(normalized_pledges, kde=True)
ax.set_title("Normalized data")
plt.show()

# TODO: Your code here!
index_of_postive_pledged=kickstarters_2017.pledged>0
positive_pledged=kickstarters_2017.pledged.loc[index_of_postive_pledged]

normalized_pledged=pd.Series(stats.boxcox(positive_pledged)[0], name='pledged', index=positive_pledged.index)
print('Original data\nPreview:\n', positive_pledged.head())
print('Minimum Value:', float(positive_pledged.min()))
print('Maximum value:', float(positive_pledged.max()))
print('_'*30)

print('Normalized data\nPreview:\n', normalized_pledged.head())
print('Minimum value: ', float(normalized_pledged.min()),
     '\nMaximum value: ', float(normalized_pledged.max()))


# plot 
ax=sns.histplot(normalized_pledged, kde=True)
ax.set_title("Normalized data")
plt.show()




##### Parsing Dates
#Load the data:
# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
landslides = pd.read_csv("../input/landslide-events/catalog.csv")

# set seed for reproducibility
np.random.seed(0)

# Check the data type of our date column
We begin by taking a look at the first five rows of the data.
landslides.head()

# print the first few rows of the date column
print(landslides['date'].head())

# Pandas uses the "object" dtype for storing various types of data types, but most often when you see a column with the dtype "object" it will have strings in it.


# check the data type of our date column
landslides['date'].dtype 

# convert date columns to datetime..
:https://strftime.org/

1/17/07 has the format "%m/%d/%y"
17-1-2007 has the format "%d-%m-%Y"

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")

# print the first few rows..
landslides['date_parsed'].head()
Name: date_parsed, dtype: datetime64[ns]
# for handling multiple date formats:
landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)

# Why don't you always use infer_datetime_format = True? 
There are two big reasons not to always have pandas guess the time format. The first is that pandas won't always been able to figure out the correct date format, especially if someone has gotten creative with data entry. The second is that it's much slower than specifying the exact format of the dates.

## Select the day of the month:
day_of_month_landslides=landslides['date_parsed'].dt.day
day_of_month_landslides.head()
o/p:
1    14.0
2    15.0
Name: date_parsed, dtype: float64

## Plot the day of month to check the date parsing..
#remove na's
day_of_month_landslides=day_of_month_landslides.dropna()

#plot the day of the month.
sns.distplot(day_of_month_landslides, kde=False, bins=31)

## Setup
from learntools.core import binder
binder.bind(globals())
from learntools.data_cleaning.ex3 import *
print("Setup Complete")

# Get our environment set up.
# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
# set seed for reproducibility
np.random.seed(0)

## #######################      CHARACTER ENCODING      #############################
## load the modules, setups and all
import pandas as pd
import numpy as np
import charset_normalizer

np.random.seed(0)

## Encoding:
character encodings are specific sets of rules for mapping from raw binary byte string (ex: 0110100001101001) to characters that make up human-readable text (ex: "hi"), 
There are many different encodings, and if you tried to read in text with a different encoding than the one it was originally written in, you ended up with scrambled text called "mojibake" (said like mo-gee-bah-kay). Here's an example of mojibake:

æ–‡å—åŒ–ã??
You might also end up with a "unknown" characters. There are what gets printed when there's no mapping between a particular byte and a character in the encoding you're using to read your byte string in and they look like this:

����������

Character encoding mismatches are less common today than they used to be, but it's definitely still a problem. There are lots of different character encodings, but the main one you need to know is UTF-8.

**UTF-8 is the standard text encoding. All Python code is in UTF-8 and, ideally, all your data should be as well. It's when things aren't in UTF-8 that you run into trouble.**

### start with a string:
before="This is euro sumbol: €"
type(before)

The other data is the bytes data type, which is a sequence of integers. You can convert a string into bytes by specifying which encoding it's in:

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors="replace")

# check the type
type(after)

you'll see that it has a b in front of it, and then maybe some text after. That's because bytes are printed out as if they were characters encoded in ASCII. (ASCII is an older character encoding that doesn't really work for writing any language other than English.) Here you can see that our euro symbol has been replaced with some mojibake that looks like "\xe2\x82\xac" when it's printed as if it were an ASCII string.

## take a look at what the bytes look like
after
o/p: b'This is the euro symbol: \xe2\x82\xac'


# convert it back to utf-8
print(after.decode("utf-8"))

# try to decode our bytes with the ascii encoding
print(after.decode("ascii"))

#### Practice:::
# setup..
from learntools.core import binder
binder.bind(globals())
from learntools.data_cleaning.ex4 import *


print("Setup Complete")

# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(




### Reading in files with encoding problems:

# try to read in a file not in UTF-8
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
//will give errors..

# look at the first ten thousand bytes to guess the character encoding
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    result = charset_normalizer.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
o/p: {'encoding': 'utf-8', 'language': 'English', 'confidence': 1.0}


# read in the file with the encoding detected by charset_normalizer
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')

# look at the first few lines
kickstarter_2016.head()



## What if the encoding charset_normalizer guesses isn't right?
Since charset_normalizer is basically just a fancy guesser, sometimes it will guess the wrong encoding. One thing you can try is looking at more or less of the file and seeing if you get a different result and then try that.

## Saving your file with UTF-8 encoding..

# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")




### Practice::
#setup:

from learntools.core import binder
binder.bind(globals())
from learntools.data_cleaning.ex4 import *
print("Setup Complete")

# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import charset_normalizer

# set seed for reproducibility
np.random.seed(0)


# What are encodings?
sample_entry = b'\xa7A\xa6n'
print(sample_entry)
print('data type:', type(sample_entry))

o/p:
b'\xa7A\xa6n'
data type: <class 'bytes'>


Use the next code cell to create a variable new_entry that changes the encoding from "big5-tw" to "utf-8". new_entry should have the bytes datatype.

before = sample_entry.decode("big5-tw", errors="replace")#____
new_entry=before.encode("utf-8")
# Check your answer
q1.check()


## Reading in files with encoding problems
Use the code cell below to read in this file at path "../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv".

Figure out what the correct encoding should be and read in the file to a DataFrame police_killings.
# TODO: Load in the DataFrame correctly.
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding='windows-1252') #____

# Check your answer
q2.check()
