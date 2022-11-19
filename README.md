# Maternal Health Risk Predictor

## Authors 

- Lennon Au-Yeung
- Chenyang Wang
- Shirley Zhang

This data analysis project was created in fulfillment of the team project requirements for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia. 

## Project Proposal

### Introduction and Questions

Maternal mortality is a large risk in lower and lower middle-income countries, with about 810 women dying from preventable pregnancy-related causes each day (WHO, 2019). Often, there is a lack of information about the woman's health during pregnancy, making it difficult to monitor their status and determine whether they may be at risk of complications (Ahmed and Kashem, 2020). A potential solution to this issue is through using the 'Internet of Things (IoT)', or physical sensors which can monitor and report different health metrics of a patient to their health care provider. Medical professionals can then analyze this information to determine whether a patient may be at risk. 

For this project, we aim to answer the question: "Can we use data analysis methods to predict the risk level of a patient during pregnancy (low, mid, or high) given a number of metrics describing their health profile?" This is an important question to explore given that human resources are low in lower income countries, and non-human dependent classification methods can help provide this information to more individuals. Furthermore, classifying a patient's risk level through data-driven methods may be advantageous over traditional methods which may involve levels of subjectivity. 

IoT sensors can collect a diverse range of health metrics, however not all of them may be useful in predicting whether a patient is at risk of adverse health outcomes. Thus, we also hope to use data analysis methods to infer whether some metrics may be more important in determining maternal health risk levels than others. 

### Data

Data used in this study was collected between 2018 and 2020, through six hospitals and maternity clinics in rural areas of Bangladesh (Ahmed and Kashem, 2020). Patients wore sensing devices which collected health data such as temperature and heart rate. The risk factor of each patient was determined through following a guideline based on previous research and consultation with medical professionals. 

The full data set was sourced from the UCI Machine Learning Repository (Dua and Graff 2017), and can be found [https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk+Data+Set](here). A .csv format of the data can be directly downloaded using [https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv](this link). The data can be attributed to Marzia Ahmed (Daffodil International University, Dhaka, Bangladesh) and Mohammod Kashem (Dhaka University of Science and Technology, Gazipur, Bangladesh) (Ahmed and Kashem, 2020).  

The data set contains six features describing a patient's health profile, including `age`, `SystolicBP` (systolic blood pressure in mmHG), `DiastolicBP` (diastolic blood pressure in mmHG), `BS` (blood glucose levels in mmol/L), `BodyTemp` (body temperature in Fahrenheit), and `HeartRate` (heart rate in beats per minute). There are 1014 instances in total, with each row corresponding to one patient. Finally, the data contains the attribute `RiskLevel`, corresponding to a medical expert's determination of whether the patient is at low, mid, or high risk (Ahmed et al., 2020). 


### Exploratory Data Analysis (EDA)

Split the data set into train data set and test data set

a. Table: 

  Distribution of 6 attributes and target class in the train data set
  
b. EDA figure: 

  i. Distribution of all 7 attributes separated/colored by target class (density distribution?) 
  
  ii. Pairwise scatter plots for all the predictors
  
### Plan of Analyzing the Data

a. Our project is a classification problem. 

b. Model we will try: 

  i. k-NN
  
  ii. Other models to try later: 
  
    1. Decision trees 
    
    2. SVMs 
    
    3. Logistic regression for classification 

### Share the Results as Tables and Figures (Results) 

The results will be shared in a Jupyter Notebook with tables and figures included and the results is reproducible.


### Usage 



## License

The maternal health risk predictor materials here are licensed under the MIT License and Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0). If re-using/re-mixing please provide attribution and link to this webpage.


### Attributions 


## References 

Ahmed, M., Kashem, M.A., Rahman, M., Khatun, S. (2020). Review and Analysis of Risk Factor of Maternal Health in Remote Area Using the Internet of Things (IoT). In: , *et al.* InECCE2019. Lecture Notes in Electrical Engineering, vol 632. Springer, Singapore. https://doi.org/10.1007/978-981-15-2317-5_30

Ahmed, M. and Kashem, M. A. (2020). IoT Based Risk Level Prediction Model For Maternal Health Care In The Context Of Bangladesh. In: 2nd International Conference on Sustainable Technologies for Industry 4.0 (STI), 2020, pp. 1-6. https://doi.org/10.1109/STI50764.2020.9350320

Dua, Dheeru, and Casey Graff. (2017). “UCI Machine Learning Repository.” University of California, Irvine, School of Information; Computer Sciences. http://archive.ics.uci.edu/ml.

WHO. (2019). "Maternal mortality". World Health Organization. https://www.who.int/news-room/fact-sheets/detail/maternal-mortality