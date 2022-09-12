import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

records = [['Beer', 'Nuts', 'Diaper'],
          ['Beer', 'Coffee', 'Diaper'],
          ['Beer', 'Diaper', 'Eggs'],
          ['Beer', 'Nuts', 'Eggs', 'Milk'],
          ['Nuts', 'Coffee', 'Diaper', 'Eggs', 'Milk']]

frequent_itemsets = apriori(records, min_support=0.3)