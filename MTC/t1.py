from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer, LabelEncoder, OrdinalEncoder
import statsmodels.api as sm

path = "C:/Users/13323/Desktop/task1.2.xlsx"
df = pd.read_excel(path, sheet_name="logistic-得分")

# 2.划分特征变量与目标变量
X = df.drop(columns='得分')
X = X.drop(columns='y-得分')
print(X.columns)
y = df['得分']
# ct_y = ColumnTransformer([('onehot', OneHotEncoder(sparse=False), ["得分"])])
ct_x = ColumnTransformer([('onehot', OneHotEncoder(sparse=False), X.columns[3:].tolist())],
                         remainder='passthrough')
a = ct_x.fit_transform(X)
print(a.shape)
X_t = pd.DataFrame(ct_x.fit_transform(X),columns=X.columns[3:].tolist()+X.columns[0:2].tolist())

# y_t = ct_y.fit_transform(y)
# # 3.划分数据集与测试集
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4.模型搭建
from sklearn.linear_model import LogisticRegression

model = sm.Logit(X_t, y)
# model = LogisticRegression(solver="liblinear")
# result = model.fit(X_t, y)
result = model.fit()
print(result.summary())

