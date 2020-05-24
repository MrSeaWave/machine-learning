import pandas as pd

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']])
print(df)

# loc函数主要通过行标签索引行数据
print("loc[0]")
print(df.loc[0])
print("loc[0:1]")
print(df.loc[0: 1])
print(" loc[0,1]")
print(df.loc[0, 1])

# iloc 主要是通过行号获取行数据，划重点，序号！序号！序号！默认是前闭后开
print("iloc[0:1]")
print(df.iloc[0:1])
print("iloc[0:1,2:3]")
print(df.iloc[0:1,2:3])
