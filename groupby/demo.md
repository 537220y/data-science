# Titanic Data Science Solutions

This notebook is a companion to the *Data Science Solutions* book and walks through a typical Kaggle competition workflow using the Titanic dataset.

## Workflow Overview

### Workflow Stages
1. Question or problem definition.
2. Acquire training and testing data.
3. Wrangle, prepare, and cleanse the data.
4. Analyze, identify patterns, and explore the data.
5. Model, predict, and solve the problem.
6. Visualize, report, and present the findings.
7. Supply or submit the results.

The sequence above is flexible—we often merge, skip, or repeat stages depending on new insights.

### Workflow Goals
- **Classifying**: understand how features correlate with survival.
- **Correlating**: measure statistical relationships between features and the target.
- **Converting**: transform categorical features into numerical representations for modeling.
- **Completing**: impute missing values so every model receives a full dataset.
- **Correcting**: remove noisy, duplicated, or error-prone features/samples.
- **Creating**: engineer additional features (e.g., Age bands, family counts, fare ranges).
- **Charting**: choose visualizations that best communicate each insight.

### Baseline Domain Assumptions
- Women were more likely to survive than men.
- Children (younger passengers) had higher survival odds.
- Upper-class passengers (Pclass = 1) were advantaged.

### Refactor Release 2017-01-29
#### User Comments
- Combine training and test sets when engineering shared features (e.g., Title extraction).
- Nearly 30% of passengers had siblings and/or spouses aboard.
- Interpret logistic regression coefficients correctly.

#### Porting Issues
- Specify plot dimensions explicitly and ensure the legend is visible.

#### Best Practices
- Run feature-correlation analysis early.
- Favor multiple simple plots over dense overlays for readability.

## Environment Setup
```python
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
```

## Acquire Data
```python
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
```

## Analyze by Describing Data
### Which features are available in the dataset?
```python
print(train_df.columns.values)
```
```text
['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']
```

### Which features are categorical?
- Survived, Sex, and Embarked are categorical (Embarked is nominal, Pclass is ordinal).

### Which features are numerical?
- Continuous: Age, Fare.
- Discrete: SibSp, Parch.

### Which features are mixed data types?
- Ticket contains numeric and alphanumeric values.
- Cabin mixes letters and numbers.

### Which features may contain errors or typos?
- Name includes titles, parentheses, and quotes, so it is noisy and non-standard.

### Which features contain blank, null, or empty values?
- Cabin > Age > Embarked contain the most nulls in the training set.
- Cabin and Age contain nulls in the test set as well.

### What are the data types for various features?
```python
train_df.info()
print('_' * 40)
test_df.info()
```
```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
________________________________________
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
```

### What is the distribution of numerical feature values?
```python
train_df.describe()
```
```text
       PassengerId    Survived      Pclass         Age       SibSp
count   891.000000  891.000000  891.000000  714.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008
std     257.353842    0.486592    0.836071   14.526497    1.102743
min       1.000000    0.000000    1.000000    0.420000    0.000000
25%     223.500000    0.000000    2.000000   20.125000    0.000000
50%     446.000000    0.000000    3.000000   28.000000    0.000000
75%     668.500000    1.000000    3.000000   38.000000    1.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000

            Parch        Fare
count  891.000000  891.000000
mean     0.381594   32.204208
std      0.806057   49.693429
min      0.000000    0.000000
25%      0.000000    7.910400
50%      0.000000   14.454200
75%      0.000000   31.000000
max      6.000000  512.329200
```

Key highlights:
- 38% of passengers survived.
- >75% traveled without parents/children.
- ~30% traveled with siblings/spouses.
- Fares vary widely (up to $512).
- Few elderly passengers.

### What is the distribution of categorical features?
```python
train_df.describe(include=['O'])
```
```text
                      Name   Sex Ticket Cabin Embarked
count                   891   891    891   204      889
unique                  891     2    681   147        3
top     Panula, Master. Juha  male   1601 C23 C25 C27        S
freq                      1   577      7     4      644
```

### Assumptions Based on Data Analysis
- **Correlating**: measure how each feature correlates with survival.
- **Completing**: fill missing Age and Embarked values.
- **Correcting**: drop Ticket (many duplicates), Cabin (mostly null), PassengerId, and possibly Name.
- **Creating**: engineer Family, Title, Age bands, Fare ranges.
- **Classifying**: validate the baseline domain assumptions listed earlier.

## Analyze by Pivoting Features
Only complete categorical/ordinal/discrete features are used in this stage.

### Pclass vs Survived
```python
train_df[['Pclass', 'Survived']]
    .groupby('Pclass', as_index=False)
    .mean()
    .sort_values(by='Survived', ascending=False)
```
```text
   Pclass  Survived
0       1  0.629630
1       2  0.472826
2       3  0.242363
```
Pclass=1 correlates strongly with survival.

### Sex vs Survived
```python
train_df[['Sex', 'Survived']]
    .groupby('Sex', as_index=False)
    .mean()
    .sort_values(by='Survived', ascending=False)
```
```text
      Sex  Survived
0  female  0.742038
1    male  0.188908
```
Females have a much higher survival rate.

### SibSp vs Survived
```python
train_df[['SibSp', 'Survived']]
    .groupby('SibSp', as_index=False)
    .mean()
    .sort_values(by='Survived', ascending=False)
```
```text
   SibSp  Survived
1      1  0.535885
2      2  0.464286
0      0  0.345395
3      3  0.250000
4      4  0.166667
5      5  0.000000
6      8  0.000000
```
SibSp correlation is non-linear, suggesting a derived feature may work better.

### Parch vs Survived
```python
train_df[['Parch', 'Survived']]
    .groupby('Parch', as_index=False)
    .mean()
    .sort_values(by='Survived', ascending=False)
```
```text
   Parch  Survived
3      3  0.600000
1      1  0.550847
2      2  0.500000
0      0  0.343658
5      5  0.200000
4      4  0.000000
6      6  0.000000
```

## Analyze by Visualizing Data
Graphs validate assumptions and highlight relationships.

### Correlating Numerical Features
```python
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
g.add_legend()
```
Observations:
- Infants (Age ≤ 4) had high survival.
- Many passengers aged 15–35 did not survive.
- Oldest passenger (Age = 80) survived.

Implications: keep Age, fill missing ages, and consider Age banding.

### Correlating Numerical and Ordinal Features
```python
grid = sns.FacetGrid(train_df, row='Pclass', col='Survived', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
```
Observations:
- Pclass=3 had the most passengers yet lowest survival.
- Infants in Pclass 2 and 3 mostly survived.
- Pclass affects the Age distribution.

### Correlating Categorical Features
```python
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
```
Observations:
- Female passengers had markedly better survival.
- Embarked interacts with Pclass and Sex.

### Correlating Categorical and Numerical Features
```python
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)
grid.add_legend()
```
Observations:
- Higher fares correlate with survival.
- Embarkation port influences the Fare-Survival relationship.

## Wrangle Data
Execute the correcting, creating, and completing goals based on the insights above.

### Drop Ticket and Cabin
```python
print('Before', train_df.shape, test_df.shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
print('After', train_df.shape, test_df.shape)
```

### Extract Title from Name
```python
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
```
```text
Sex     female  male
Title                
Capt         0     1
Col          0     2
Countess     1     0
Don          0     1
Dr           1     6
Jonkheer     0     1
Lady         1     0
Major        0     2
Master       0    40
Miss       182     0
Mlle         2     0
Mme          1     0
Mr           0   517
Mrs        125     0
Ms           1     0
Rev          0     6
Sir          0     1
```
Simplify and map titles:
```python
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare'
    )
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping).fillna(0)
```
Drop the original Name feature and PassengerId from the training set:
```python
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
```

### Convert Sex to Numeric
```python
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
```

### Complete Age
Use median age per Sex/Pclass group:
```python
guess_ages = np.zeros((2, 3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[
                (dataset['Age'].isnull()) &
                (dataset['Sex'] == i) &
                (dataset['Pclass'] == j + 1),
                'Age'
            ] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)
```
Create Age bands and convert to ordinals:
```python
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']]
    .groupby('AgeBand', as_index=False)
    .mean()
    .sort_values(by='AgeBand')
```
```text
          AgeBand  Survived
0  (-0.08, 16.0]  0.550000
1   (16.0, 32.0]  0.337374
2   (32.0, 48.0]  0.412037
3   (48.0, 64.0]  0.434783
4    (64.0, 80.0] 0.090909
```
```python
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
```

### Create FamilySize and IsAlone
```python
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']]
    .groupby('FamilySize', as_index=False)
    .mean()
    .sort_values(by='Survived', ascending=False)
```
```text
   FamilySize  Survived
3           4  0.724138
2           3  0.578431
1           2  0.552795
6           7  0.333333
0           1  0.303538
4           5  0.200000
5           6  0.136364
7           8  0.000000
8          11  0.000000
```
```python
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
```

### Create Age*Class
```python
for dataset in combine:
    dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']
```

### Complete Embarked and Convert to Numeric
```python
freq_port = train_df['Embarked'].dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
```

### Complete and Band Fare
```python
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].dropna().median())

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']]
    .groupby('FareBand', as_index=False)
    .mean()
    .sort_values(by='FareBand')
```
```text
           FareBand  Survived
0    (-0.001, 7.91]  0.197309
1   (7.91, 14.454]  0.303571
2    (14.454, 31.0] 0.454955
3  (31.0, 512.329]  0.581081
```
```python
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
```

Sample of the engineered training data:
```python
train_df.head()
```
```text
   Survived  Pclass  Sex  Age  Fare  Embarked  Title  IsAlone  Age*Class
0         0       3    0    1     0         0      1        0          3
1         1       1    1    2     3         1      3        0          2
2         1       3    1    1     1         0      2        1          3
3         1       1    1    2     3         0      3        0          2
4         0       3    0    2     1         0      1        1          6
```

## Model, Predict, and Solve
Split data and evaluate several supervised-learning algorithms.
```python
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
```

### Logistic Regression
```python
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
```
Accuracy: **80.36%**.

Feature coefficients highlight importance of Sex (positive), Title, Age, Embarked, and the negative impact of higher Pclass and Age*Class.

### Support Vector Machines
```python
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
```
Accuracy: **83.84%**.

### k-Nearest Neighbors
```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
```
Accuracy: **84.74%**.

### Gaussian Naive Bayes
```python
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
```
Accuracy: **72.28%**.

### Perceptron
```python
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
```
Accuracy: **78.00%**.

### Linear SVC
```python
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
```
Accuracy: **79.12%**.

### Stochastic Gradient Descent
```python
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
```
Accuracy: **78.56%**.

### Decision Tree
```python
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
```
Accuracy: **86.76%**.

### Random Forest
```python
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
```
Accuracy: **86.76%** (preferred to mitigate individual tree overfitting).

## Model Evaluation and Submission
```python
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]
})
models.sort_values(by='Score', ascending=False)
```
```text
                       Model  Score
3             Random Forest  86.76
8              Decision Tree 86.76
1                        KNN 84.74
0  Support Vector Machines   83.84
2          Logistic Regression 80.36
7                  Linear SVC 79.12
6  Stochastic Gradient Decent 78.56
5                  Perceptron 78.00
4                 Naive Bayes 72.28
```
```python
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': Y_pred
})
# submission.to_csv('../output/submission.csv', index=False)
```
The Kaggle submission scored 3,883 / 6,082 during the competition—respectable for a first pass.

## References
- *A Journey through Titanic*.
- *Getting Started with Pandas: Kaggle's Titanic Competition*.
- *Titanic Best Working Classifier*.
