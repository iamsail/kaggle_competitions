#!usr/bin/python
# -*- coding:utf-8 -*-
"""visualize the test data to get feature

@author: Sail
@contact: 865605793@qq.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

data_train = pd.read_csv("./data/train.csv")

# 乘客各属性分布
def visualize_passengers_property():
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
    data_train.Survived.value_counts().plot(kind='bar')# 柱状图
    plt.title(u"获救情况 (1为获救)") # 标题
    plt.ylabel(u"人数")

    plt.subplot2grid((2,3),(0,1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"乘客等级分布")

    plt.subplot2grid((2,3),(0,2))
    plt.scatter(data_train.Survived, data_train.Age)
    plt.ylabel(u"年龄")                         # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布 (1为获救)")


    plt.subplot2grid((2,3),(1,0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")# plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


    plt.subplot2grid((2,3),(1,2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")
    plt.show()


## 看各乘客等级的获救情况
def visualize_passengers_survived_by_pclass():
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u'各乘客等级的获救情况')
    plt.xlabel(u'乘客等级')
    plt.ylabel(u'人数')
    plt.show()


# 各个舱级别下各性别的获救情况
def visualize_passengers_survived_by_sex_and_cabin():
    fig = plt.figure()
    fig.set(alpha=0.65)
    plt.title(u"根据舱等级和性别的获救情况")

    ax1 = fig.add_subplot(141)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='female highclass', color='#FA2479')
    ax1.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"女性/高级舱"], loc='best')

    ax2 = fig.add_subplot(142, sharey = ax1)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female highclass', color='pink')
    ax1.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"女性/高级级舱"], loc='best')

    ax3=fig.add_subplot(143, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
    ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/高级舱"], loc='best')

    ax4=fig.add_subplot(144, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
    ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/低级舱"], loc='best')

    plt.show()




#各登船港口的获救情况
def visualize_passengers_survived_by_embarked():
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u'各登陆港口乘客的获救情况')
    plt.xlabel(u'登陆港口')
    plt.ylabel(u'人数')

    plt.show()


# 堂兄弟/妹 or 孩子/父母有几人，对是否获救的影响。
def whether_effect_survived(field):
    g = data_train.groupby([field, 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)



# #ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
# #cabin只有204个乘客有值，我们先看看它的一个分布
# data_train.Cabin.value_counts()
# print(data_train.Cabin.value_counts())


# 有无Cabin信息这个粗粒度上看看Survived的情况好了
def visualize_passengers_survived_by_cabin_exist():
    fig = plt.figure()
    fig.set(alpha = 1)
    Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
    Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
    df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
    df.plot(kind='bar', stacked=True)
    plt.title(u"按Cabin有无看获救情况")
    plt.xlabel(u"Cabin有无")
    plt.ylabel(u"人数")
    plt.show()


def visual_controller():
    visualize_passengers_property()
    visualize_passengers_survived_by_pclass()
    visualize_passengers_survived_by_sex_and_cabin()
    visualize_passengers_survived_by_embarked()
    whether_effect_survived('SibSp')
    whether_effect_survived('Parch')
    visualize_passengers_survived_by_cabin_exist()


if __name__ == '__main__':
    visual_controller()