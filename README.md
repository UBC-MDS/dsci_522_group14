# Maternal Health Risk Predictor

## Authors 

- Lennon Au-Yeung
- Chenyang Wang
- Shirley Zhang

(Team 14) 

This data analysis project was created in fulfillment of the team project requirements for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia. 

## Introduction and Questions

Maternal mortality is a large risk in lower and lower middle-income countries, with about 810 women dying from preventable pregnancy-related causes each day (WHO, 2019). Often, there is a lack of information about the woman's health during pregnancy, making it difficult to monitor their status and determine whether they may be at risk of complications (Ahmed and Kashem, 2020). A potential solution to this issue is through using the 'Internet of Things (IoT)', or physical sensors which can monitor and report different health metrics of a patient to their health care provider. Medical professionals can then analyze this information to determine whether a patient may be at risk. 

For this project, we aim to answer the question: 

> **"Can we use data analysis methods to predict the risk level of a patient during pregnancy (low, mid, or high) given a number of metrics describing their health profile?"** 

This is an important question to explore given that human resources are low in lower income countries, and non-human dependent classification methods can help provide this information to more individuals. Furthermore, classifying a patient's risk level through data-driven methods may be advantageous over traditional methods which may involve levels of subjectivity. 


IoT sensors can collect a diverse range of health metrics, however not all of them may be useful in predicting whether a patient is at risk of adverse health outcomes. Thus, we also hope to use data analysis methods to infer (sub-question) whether some metrics may be more important in determining maternal health risk levels than others. 


## Data

Data used in this study was collected between 2018 and 2020, through six hospitals and maternity clinics in rural areas of Bangladesh (Ahmed and Kashem, 2020). Patients wore sensing devices which collected health data such as temperature and heart rate. The risk factor of each patient was determined through following a guideline based on previous research and consultation with medical professionals. 

The full data set was sourced from the UCI Machine Learning Repository (Dua and Graff 2017), and can be found [here](https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk+Data+Set). A .csv format of the data can be directly downloaded using [this link](https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv). The data can be attributed to Marzia Ahmed (Daffodil International University, Dhaka, Bangladesh) and Mohammod Kashem (Dhaka University of Science and Technology, Gazipur, Bangladesh) (Ahmed and Kashem, 2020).  

The data set contains six features describing a patient's health profile, including `age`, `SystolicBP` (systolic blood pressure in mmHG), `DiastolicBP` (diastolic blood pressure in mmHG), `BS` (blood glucose levels in mmol/L), `BodyTemp` (body temperature in Fahrenheit), and `HeartRate` (heart rate in beats per minute). There are 1014 instances in total, with each row corresponding to one patient. Finally, the data contains the attribute `RiskLevel`, corresponding to a medical expert's determination of whether the patient is at low, mid, or high risk (Ahmed et al., 2020). 


## Exploratory Data Analysis 


Before exploration, we will shuffle and split our data into training and test sets (80% training, 20% test). We will explore the distribution of target classes (`RiskLevel`), to determine whether there is a class imbalance that needs to be accounted for. This will be done through tables displaying the class counts for the training data set. We will create a bar plot of target class to help further visualize the distribution. Furthermore, we will examine the summary table with metrics such as mean, min, and max values for our features to determine how to best preprocess our dataset. 

Next, given that all six features are continuous, we will create density distributions of each feature colored by the different target classes. This can give us a hint of whether the distribution of some features are different for different risk levels. If some features have similar distributions for all classes, it may suggest they are not helpful in predicting the target and hence can be dropped. We will also create a correlation matrix and pairwise scatter plots for all features, to examine whether some features may be correlated with one another. 

The report of our preliminary exploratory data analysis can be found here.  

## Analysis 

To answer our main question, we will employ a predictive classification model on our data set. We aim to use a $k$-NN ($k$-Nearest Neighbours) algorithm with hyperparameter optimization to select the optimal 'K' number of neighbours. We will perform cross-validation with approximately 30 folds, given that we have a relatively small amount of observations (n = 1014). We will use overall accuracy as the model evaluation metric and plot the accuracy with different values of 'K'. After selecting the best 'K', we will refit our model to our training data and evaluate using the test data. Finally, we will use overall accuracy and confusion matrices to examine how well our model performed on the test set. 

Time permitting, we will repeat our analysis with the following classification models and hyperparameter optimization: 

1. Decision trees 
2. Support Vector Machines (SVMs)
3. Logistic regression 

We will compare the results of these models to our $k$-NN model to determine whether one performs better than others. Furthermore, these models may give us information to answer our inferential sub-question of which features are more important for predicting our target class. 


To ensure reproducibility, the results will be shared in a Jupyter Notebook with tables, figures, interpretations, and corresponding code/scripts included. 

## Usage 

To replicate the analysis done in this project, you must clone the repository and run the following below. 

To download the full data set and save it as 'maternal_risk.csv' in your current working directory: 

```
python src/download_data.py --out_type='csv' --url='https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv' --out_file='maternal_risk.csv'
```

To render the exploratory data analysis file, open the `src/maternal_risk_eda.ipynb` file in jupyter lab or another IDE and run all the cells. 

## Dependencies 

Python 3.10 and Python packages:
- docopt==0.6.2
- pandas==1.5.1
- altair
- altair_saver
- requests=2.22.0

To ensure reproducibility, the results will be shared in a Jupyter Notebook with tables, figures, and corresponding code/scripts included. 

## Report 

To be added once our full analysis is complete. 

## Usage 

To replicate the analysis done in this project, you follow the steps below:

1. Install the dependencies listed under "Dependencies"

2. Clone the repository (the following shows cloning through ssh keys):

```
git clone git@github.com:UBC-MDS/maternal_health_risk_predictor.git
```

3. Move to the cloned directory

```
cd maternal_health_risk_predictor
```

4. Download the full data set and save it as 'maternal_risk.csv' under the `data/src/` directory: 

```
python src/download_data.py --out_type='csv' --url='https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv' --out_file='data/raw/maternal_risk.csv'
```

(note: change the path and filename for the option '--out_file' if you wish to save the data in a different directory)

5. To render the exploratory data analysis file, open the `src/maternal_risk_eda.ipynb` file in jupyter lab or another IDE and run all the cells. 

(more steps to be added as the project moves towards completion) 


## License

The Maternal Health Risk Predictor materials are licensed under the Creative Commons Attribution 2.5 Canada License (CC BY 2.5 CA). If re-using/re-mixing please provide attribution and link to this webpage.


Further license information can be viewed in the `LICENSE` file in the root folder of this repository.


## Attributions 

The data set is attributed to Marzia Ahmed and Mohammod Kashem (Ahmed and Kashem, 2020) as well as the UCI Machine Learning Repository (Dua and Graff 2017). 


The code of conduct file was adapted from the [Contributor Covenant][homepage], version 1.4, available at [http://contributor-covenant.org/version/1/4][version]
[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/

The contributions file was adapted from the [dplyr contributing guidelines](https://github.com/tidyverse/dplyr/blob/master/.github/CONTRIBUTING.md) and [breast cancer predictor contributing guidelines](https://github.com/ttimbers/breast_cancer_predictor/blob/master/CONTRIBUTING.md).


## References 

Ahmed, M., Kashem, M.A., Rahman, M., Khatun, S. (2020). Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT). In: , *et al.* InECCE2019. Lecture Notes in Electrical Engineering, vol 632. Springer, Singapore. https://doi.org/10.1007/978-981-15-2317-5_30

Ahmed, M. and Kashem, M. A. (2020). IoT Based Risk Level Prediction Model For Maternal Health Care In The Context Of Bangladesh. In: 2nd International Conference on Sustainable Technologies for Industry 4.0 (STI), 2020, pp. 1-6. https://doi.org/10.1109/STI50764.2020.9350320

Dua, Dheeru, and Casey Graff. (2017). “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.

WHO. (2019). "Maternal mortality". World Health Organization. https://www.who.int/news-room/fact-sheets/detail/maternal-mortality