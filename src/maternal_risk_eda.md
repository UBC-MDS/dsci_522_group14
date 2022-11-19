---
---
---

# Exploratory data analysis of the Maternal Health Risk Data Set

Team: 14

Team members: Lennon Au-Yeung, Chenyang Wang, Shirley Zhang

``` python
#import the required packages
import pandas as pd
from sklearn.model_selection import train_test_split
import altair as alt
alt.renderers.enable('mimetype')
alt.data_transformers.enable('data_server')
```

    RendererRegistry.enable('mimetype')

# Summary of the data set

The data set used in this project consists of health information collected from the rural areas of Bangladesh created by Dr. Marzia Ahmed at the Daffodil International University. It was sourced from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk+Data+Set), specifically [this file](https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv). Each row in the data set consists of responsible and significant risk factors for maternal mortality (e.g., blood glucose levels, blood sugar and body temperature) and maternity risk level (high, mid, or low). Information was extracted through an IoT (Internet of Things) based risk monitoring system. There are 1014 observations in the data set, and 6 features. There are 0 observations with missing values in the data set.

``` python
#loading the data set
maternal_risk_df = pd.read_csv('../data/raw/maternal_risk.csv',header=1)
```

``` python
summary_df = pd.DataFrame(maternal_risk_df['RiskLevel'].value_counts()).rename(columns={'RiskLevel':''}).T
summary_df
```

<div>

```{=html}
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
```
|     | low risk | mid risk | high risk |
|-----|----------|----------|-----------|
|     | 406      | 336      | 272       |

</div>

# Split data set into training and test splits

Before proceeding further, we will split the data such that 80% of observations are in the training and 20% of observations are in the test set.

``` python
#train_test_split
train_df, test_df = train_test_split(maternal_risk_df, test_size=0.20, random_state=123)
```

We also need to check that there are no null values in the data set.

``` python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 811 entries, 375 to 510
    Data columns (total 7 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Age          811 non-null    int64  
     1   SystolicBP   811 non-null    int64  
     2   DiastolicBP  811 non-null    int64  
     3   BS           811 non-null    float64
     4   BodyTemp     811 non-null    float64
     5   HeartRate    811 non-null    int64  
     6   RiskLevel    811 non-null    object 
    dtypes: float64(2), int64(4), object(1)
    memory usage: 50.7+ KB

Table 1. Number of non-null values for each column.

``` python
train_df.describe()
```

<div>

```{=html}
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
```
|       | Age        | SystolicBP | DiastolicBP | BS         | BodyTemp   | HeartRate  |
|----------|----------|----------|----------|----------|----------|----------|
| count | 811.000000 | 811.000000 | 811.000000  | 811.000000 | 811.000000 | 811.000000 |
| mean  | 29.574599  | 112.933416 | 76.262639   | 8.659211   | 98.680641  | 74.373613  |
| std   | 13.287246  | 18.334896  | 13.764557   | 3.223935   | 1.379661   | 7.908723   |
| min   | 10.000000  | 70.000000  | 49.000000   | 6.000000   | 98.000000  | 7.000000   |
| 25%   | 19.000000  | 95.000000  | 65.000000   | 6.900000   | 98.000000  | 70.000000  |
| 50%   | 25.000000  | 120.000000 | 80.000000   | 7.500000   | 98.000000  | 76.000000  |
| 75%   | 37.500000  | 120.000000 | 90.000000   | 7.950000   | 98.000000  | 80.000000  |
| max   | 66.000000  | 160.000000 | 100.000000  | 19.000000  | 103.000000 | 90.000000  |

</div>

Table 2. Statistics of each numeric columns.

Now we will explore the distribution of the classes (low risk, mid risk, high risk):

``` python
train_class_counts = pd.DataFrame(train_df['RiskLevel'].value_counts()).rename(columns={'RiskLevel':'train'})
test_class_counts = pd.DataFrame(test_df['RiskLevel'].value_counts()).rename(columns={'RiskLevel':'test'})

train_test_class_counts = pd.concat([train_class_counts, test_class_counts], axis=1)
train_test_class_counts
```

<div>

```{=html}
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
```
|           | train | test |
|-----------|-------|------|
| low risk  | 325   | 81   |
| mid risk  | 274   | 62   |
| high risk | 212   | 60   |

</div>

Table 3. Counts of observation for each class in both test and train.

``` python
train_df['RiskLevel'].value_counts().plot(kind='bar')
```

    <AxesSubplot:>

![png](maternal_risk_eda_figures/output_18_1.png)

Figure 1. Counts of observation for each class in train data set.

From the figure, we can see that there is a minor class imbalance, but it is not so great that immediate action needs to be taken. During hyperparameter optimization for improving model performance, we can further evaluate whether a balanced class weight will improve model performance or not.

# Exploratory analysis on the training data set

``` python
X_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
```

``` python
def display(i):
    graph = alt.Chart(train_df).transform_density(
    i,groupby=['RiskLevel'],
    as_=[ i, 'density']).mark_line().encode(
    x = (i),
    y='density:Q',color = 'RiskLevel').properties(width=200,height=200)
    return graph
```

``` python
Age = display('Age')
SystolicBP = display('SystolicBP')
DiastolicBP = display('DiastolicBP')
BS = display('BS')
BodyTemp = display('BodyTemp')
HeartRate = display('HeartRate')
```

To see whether the predictors might be useful to predict the risk level, we plotted the distributions of each predictor from the training data and coloured the distribution by class (high risk: blue, mid risk: red and low risk: orange).

``` python
((Age | SystolicBP | DiastolicBP) & (BS | BodyTemp | HeartRate)).properties(title='Distribution of predictors for each Risk Level')
```

![png](maternal_risk_eda_figures/output_26_0.png)

Figure 2. Distribution of training set predictors for high risk, mid risk and low risk

From the figure above, we can see that in SystolicBP and DiastolicBP, observations with high value in both of the categories are mostly associated with high maternity risk level. We can see that low blood glucose level is more associated with mid and low maternity risk level, while the density of high risk level in blood glucose level is the same throughout. For other predictors, we can see that the distribution is similar across all three risk levels.

To explore whether there is any interesting relationship between the predictors, we have plotted a correlation matrix and pairwise scatter plots for all the predictors.

``` python
train_df.corr('spearman').style.background_gradient()
```

```{=html}
<style type="text/css">
#T_df9df_row0_col0, #T_df9df_row1_col1, #T_df9df_row2_col2, #T_df9df_row3_col3, #T_df9df_row4_col4, #T_df9df_row5_col5 {
  background-color: #023858;
  color: #f1f1f1;
}
#T_df9df_row0_col1 {
  background-color: #4697c4;
  color: #f1f1f1;
}
#T_df9df_row0_col2 {
  background-color: #5a9ec9;
  color: #f1f1f1;
}
#T_df9df_row0_col3 {
  background-color: #b5c4df;
  color: #000000;
}
#T_df9df_row0_col4, #T_df9df_row2_col5, #T_df9df_row4_col0, #T_df9df_row4_col1, #T_df9df_row4_col2, #T_df9df_row4_col3 {
  background-color: #fff7fb;
  color: #000000;
}
#T_df9df_row0_col5 {
  background-color: #ede8f3;
  color: #000000;
}
#T_df9df_row1_col0 {
  background-color: #4295c3;
  color: #f1f1f1;
}
#T_df9df_row1_col2, #T_df9df_row2_col1 {
  background-color: #0566a0;
  color: #f1f1f1;
}
#T_df9df_row1_col3 {
  background-color: #cdd0e5;
  color: #000000;
}
#T_df9df_row1_col4 {
  background-color: #fcf4fa;
  color: #000000;
}
#T_df9df_row1_col5, #T_df9df_row2_col4 {
  background-color: #fbf3f9;
  color: #000000;
}
#T_df9df_row2_col0 {
  background-color: #529bc7;
  color: #f1f1f1;
}
#T_df9df_row2_col3 {
  background-color: #b7c5df;
  color: #000000;
}
#T_df9df_row3_col0 {
  background-color: #7bacd1;
  color: #f1f1f1;
}
#T_df9df_row3_col1 {
  background-color: #96b6d7;
  color: #000000;
}
#T_df9df_row3_col2 {
  background-color: #83afd3;
  color: #f1f1f1;
}
#T_df9df_row3_col4 {
  background-color: #d5d5e8;
  color: #000000;
}
#T_df9df_row3_col5 {
  background-color: #e2dfee;
  color: #000000;
}
#T_df9df_row4_col5 {
  background-color: #eae6f1;
  color: #000000;
}
#T_df9df_row5_col0 {
  background-color: #c1cae2;
  color: #000000;
}
#T_df9df_row5_col1 {
  background-color: #d9d8ea;
  color: #000000;
}
#T_df9df_row5_col2 {
  background-color: #e0deed;
  color: #000000;
}
#T_df9df_row5_col3 {
  background-color: #e9e5f1;
  color: #000000;
}
#T_df9df_row5_col4 {
  background-color: #bdc8e1;
  color: #000000;
}
</style>
```
|             | Age       | SystolicBP | DiastolicBP | BS        | BodyTemp  | HeartRate |
|----------|----------|----------|----------|----------|----------|----------|
| Age         | 1.000000  | 0.473240   | 0.430211    | 0.318194  | -0.317851 | 0.072938  |
| SystolicBP  | 0.473240  | 1.000000   | 0.752019    | 0.249512  | -0.288564 | -0.021214 |
| DiastolicBP | 0.430211  | 0.752019   | 1.000000    | 0.314118  | -0.278765 | -0.052615 |
| BS          | 0.318194  | 0.249512   | 0.314118    | 1.000000  | -0.015323 | 0.124527  |
| BodyTemp    | -0.317851 | -0.288564  | -0.278765   | -0.015323 | 1.000000  | 0.088351  |
| HeartRate   | 0.072938  | -0.021214  | -0.052615   | 0.124527  | 0.088351  | 1.000000  |

Table 4. Correlation matrix of all predictors

``` python
alt.Chart(train_df).mark_point(opacity=0.3, size=10).encode(
     alt.X(alt.repeat('row'), type='quantitative'),
     alt.Y(alt.repeat('column'), type='quantitative')
).properties(
    width=100,
    height=100
).repeat(
    column=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'],
    row=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
)
```

![png](maternal_risk_eda_figures/output_32_0.png)

Figure 3. Pairwise relationship between predictors.

From the above table and figure, we can see that the features SystolicBP and DiastolicBP have high correlation compared to other pairs of predictors, followed by the correlation between the two blood pressure levels and age. For other pairs of predictors, there are no significant correlations found.

# References

Dua, Dheeru, and Casey Graff. 2017. "UCI Machine Learning Repository." University of California, Irvine, School of Information; Computer Sciences. <http://archive.ics.uci.edu/ml>.
