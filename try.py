import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
# df = pd.read_csv('data/sampling_7w.csv')
# print(df)

df = pd.read_json('data/yelp_academic_dataset_review.json', lines=True)
df = df.sort_values('useful',False)
df = df[:200000]
df = df[['stars','text']]
print(len(df))
# df = df.sample(n=200000)
df.to_csv("data/sampling_20w_useful.csv",index=False)

# df = pd.read_csv('data/sampling_20w.csv')
# print(df[df['useful']==111])
