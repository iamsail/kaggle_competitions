#!usr/bin/python
# -*- coding:utf-8 -*-
"""the model of titanic-competition on kaggle

@author: Sail
@contact: 865605793@qq.com
"""
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from analysis_plot import visual_controller




data_train = pd.read_csv("./data/train.csv")

from sklearn.ensemble import RandomForestRegressor
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def handle_missing_value(data_train):
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    return  data_train, rfr



# 先对类目型的特征因子化
def feature_factorization(data):
    dummies_Cabin = pd.get_dummies(data['Cabin'], prefix= 'Cabin')
    dummies_Embarked = pd.get_dummies(data['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data['Pclass'], prefix= 'Pclass')
    df = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df




# 将一些变化幅度较大的特征化到[-1,1]之内
def scale_feature():
    import sklearn.preprocessing as preprocessing
    scaler = preprocessing.StandardScaler()
    age_fare = df.as_matrix(['Age','Fare'])# 返回二维array
    scaler.fit(age_fare)
    scaled_age_fare = scaler.transform(age_fare)
    df['Age_scaled'] = scaled_age_fare[:,0]
    df['Fare_scaled'] = scaled_age_fare[:,1]
    return scaler,df





# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模。
def build_model():
    from sklearn import linear_model

    # 用正则取出我们要的属性值
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)
    return clf



# 我们的”test_data”也要做和”train_data”一样的预处理啊！！
def handle_test_data():
    data_test = pd.read_csv("./data/test.csv")
    data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
    # 接着我们对test_data做和train_data中一致的特征变换
    # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

    data_test = set_Cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    test_age_fare=df_test.as_matrix(['Age','Fare'])# 返回二维array
    scaler.fit(test_age_fare)
    scaled_age_fare=scaler.transform(test_age_fare)
    df_test['Age_scaled']=scaled_age_fare[:,0]
    df_test['Fare_scaled']=scaled_age_fare[:,1]
    return df_test,data_test




# 数据可视化控制器.帮助分析
# visual_controller()
data_train, rfr = handle_missing_value(data_train)
df = feature_factorization(data_train)
scaler,df = scale_feature()
clf = build_model()
df_test,data_test = handle_test_data()






test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("./data/logistic_regression_predictions.csv", index=False)