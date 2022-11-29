# Makefile
# Shirley Zhang, November 2022

# ... description ... 
# ... example usage ... 


all : doc/final_report.Rmd


# download data 
data/raw/maternal_risk.csv : 


# preprocess data 
data/processed/test_df.csv : 

data/processed/test_df_binary.csv : 

data/processed/train_df.csv : 

data/processed/train_df_binary.csv : 


# create the figures from EDA 
src/maternal_risk_eda_figures/EDA.png : 

src/maternal_risk_eda_figures/box_plot.png : 

src/maternal_risk_eda_figures/class_distribution : 

src/maternal_risk_eda_figures/density_plot.png : 

src/maternal_risk_eda_figures/output_32_0.png : 


# create the figures from model building 
src/maternal_risk_model_figures/hyperparam_plot.png : 

src/maternal_risk_model_figures/model_comparison_table.csv : 

src/maternal_risk_model_figures/testdata_confusion_matrix.csv : 


# test model 


# create the final report 
doc/final_report.Rmd : 


clean : 