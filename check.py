import pandas as pd # Basic data manipulation

print("Lion-mod dataset size:")
data = pd.read_csv("HR Employee Attrition.csv")
# print(data.info())
print(data.shape)
print()
print("Original dataset size:")
data = pd.read_csv("yoyo.csv")
# print(data.info())
print(data.shape)
