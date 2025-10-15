import pandas as pd
df=pd.read_csv("Seed_Data.csv")
#df.describe() /quick summary or report card for your data.
#df.head()/Display the first five rows
#df.tail()/display last rows
#df.info() df.shape /Check basic info and size:
f=(df['LK'].mean())# to find mean of LK
d=(df['WK'].median())# to find median of WK
y=(df['LKG'].mode())#to find mode of LKG
print(f)
print(d)
print(y)
