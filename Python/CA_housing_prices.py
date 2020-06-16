# Task: use california census data to predict the median housing price of a block
#       in cali. a block is a population of 600-3000
# objective: get an accurate median to feed down the line to an investment analyzer 
# ================================================================================= 

import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)
# TODO Make a seperate function for this so it is generalizable to other data sets 

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32 # 2^32

def split_train_test_set_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

fetch_housing_data()
housing = load_housing_data()

"""# commented out to run rest of code. delete quotes to run
print(housing.head())
housing.info()
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

# house_hist = housing.hist(bins=50, figsize=(20,15))
# plt.show(house_hist.all()) # any() or all() need to be specified for plot to deal with ambiguity
"""
train_set, test_set = split_train_test(housing, 0.2)
#print(len(train_set))
#print(len(test_set))

# clean this up later  with explenation, all three fulfill the same goal
#housing_with_id = housing.reset_index()
#train_set, test_set = split_train_test_set_by_id(housing_with_id, 0.2, "index")

#housing_with_id["id"] = housing["longtitude"] * 1000 + housing["lattitude"]
#train_set, test_set = split_train_test_set_by_id(housing_with_id, 0.2, "index")

# creates a purely random test data using functions built into scikit-learn
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# create stratums for more representative test dataset not skewed by purely random sampling
housing["income_cat"] = pd.cut(housing["median_income"],
                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                        labels=[1, 2, 3, 4, 5])
#housing["income_cat"].hist()
#plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set  = housing.loc[test_index]


#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set)) 

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter",x="longitude",y="latitude", alpha=0.25, 
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True
             )
#plt.legend()
#plt.show()

corr_matrix = housing.corr()

#print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
pd.plotting.scatter_matrix(housing[attributes], figsize=(12,8))
#plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#plt.show()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

# ====================================================================================
# actual machine learning part!!

# clean up original data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# remove empty or missing features, three approaches
#housing.dropna(subset=["total_bedrooms"])  # option 1
#housing.drop("total_bedrooms", axis=1)     # option 2
#median = housing["total_bedrooms"].median() # option 3. requires median calculation
#housing["total_bedrooms"].fillna(median, inplace=True)

# scikit has a built in function that performs option 3
imputer = SimpleImputer(strategy="median")

# create a copy of data withouht text in order to compute median
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

#print(imputer.statistics_) #confirms medians are the same
#print(housing_num.median().values)

# replace the missing values with median using imputer 
X = imputer.transform(housing_num) # use variable more descriptive than X next time

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# in order to extract info from text, we encode text categories into numbers
housing_cat = housing[["ocean_proximity"]]
#ordinal_enc = OrdinalEncoder() # ordinal encoding is useful when the text is ordered
#housing_cat_encoded = ordinal.encoder.fit_transform(housing_cat)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#print(housing_cat_1hot) outputs sparse array, add .toarray() for dense output

#left of at page 107