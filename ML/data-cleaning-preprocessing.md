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

**If a value is missing becuase it doesn't exist (like the height of the oldest child of someone who doesn't have any children) then it doesn't make sense to try and guess what it might be. These values you probably do want to keep as NaN. On the other hand, if a value is missing because it wasn't recorded, then you can try to guess what it might have been based on the other values in that column and row. This is called imputation**


