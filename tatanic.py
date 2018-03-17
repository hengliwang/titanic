import pandas as pd
import numpy as np
import re

# 读取训练和测试数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 存储用户
PassengerId = test['PassengerId']
#显示前三行
#train.head(3);
#print(train.head(3));

#特征工程
full_data = [train, test]

# 1.姓名的长度
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# 2.游客是否有船舱
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# 3.家庭人数
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# 4.是否独自一人
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Embarked 栏空值填充为出现次数最多的S
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Fare空值填充为中值
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# 5.填充年龄中的空值
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    #处理年龄中的空值，使用均值和标准差，围绕均值随机生成数值填入
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)


# 从旅客姓名中提取title,结果分别为Mr/Miss/Mrs
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# 6.提取title
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    #one-hot mapping
    pclass = pd.get_dummies(dataset['Pclass'], prefix='Pclass')
    dataset['Pclass_1'] = pclass.Pclass_1;
    dataset['Pclass_2'] = pclass.Pclass_2;
    dataset['Pclass_3'] = pclass.Pclass_3;
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    dataset['Title'] = dataset['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }
    # we map each title
    dataset['Title'] = dataset.Title.map(Title_Dictionary)
    title = pd.get_dummies(dataset.Title)
    dataset['Royalty'] = title.Royalty;
    dataset['Officer'] = title.Officer
    dataset['Miss'] = title.Miss
    dataset['Mrs'] = title.Mrs
    dataset['Mr'] = title.Mr
    dataset['Master'] = title.Master
    # Mapping Embarked
    #dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    embarked = pd.get_dummies(dataset['Embarked'], prefix='Embarked')
    dataset['Embarked_S'] = embarked.Embarked_S;
    dataset['Embarked_C'] = embarked.Embarked_C;
    dataset['Embarked_Q'] = embarked.Embarked_Q;
    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']=5;
    #Mapping Cabin
    dataset['Cabin'] = dataset.Cabin.fillna('U')

    # mapping each Cabin value with the cabin letter
    dataset['Cabin'] = dataset['Cabin'].map(lambda c: c[0])
    # dummy encoding ...
    cabin = pd.get_dummies(dataset['Cabin'], prefix='Cabin')
    dataset['Cabin_A'] = cabin.Cabin_A;
    dataset['Cabin_B'] = cabin.Cabin_B;
    dataset['Cabin_C'] = cabin.Cabin_C;
    dataset['Cabin_D'] = cabin.Cabin_D;
    dataset['Cabin_E'] = cabin.Cabin_E;
    dataset['Cabin_F'] = cabin.Cabin_F;
    dataset['Cabin_G'] = cabin.Cabin_G;
    #dataset['Cabin_T'] = cabin.Cabin_T;
    dataset['Cabin_U'] = cabin.Cabin_U;

drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Pclass', 'Title', 'Embarked']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test = test.drop(drop_elements, axis = 1)
#y = train['Survived'];
#train,test,y_train,y_test = train_test_split(train,y,test_size=0.33,random_state=42);


result_train = pd.DataFrame(train,index=None)
result_train.to_csv('train_result.csv',index=False)

result_test = pd.DataFrame(test,index=None)
result_test.to_csv('test_result.csv',index=False)