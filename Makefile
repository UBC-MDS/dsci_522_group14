# Maternal Health Risk Predictor Makefile Pipeline
# Author: Shirley Zhang, Lennon Au-Yeung
# Date: 2022-11-29

# ... description ... 
# ... example usage ... 


all : doc/final_report.md


# download data 
data/raw/maternal_risk.csv : src/download_data.py
	python src/download_data.py --out_type='csv' --url='https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv' --out_file='data/raw/maternal_risk.csv'


# preprocess data 
data/processed/test_df.csv data/processed/test_df_binary.csv data/processed/train_df.csv data/processed/train_df_binary.csv : data/raw/maternal_risk.csv src/pre_processing.py 
	python src/pre_processing.py --data_location='data/raw/maternal_risk.csv' --output_location='data/processed/'


# create the figures from EDA 
src/maternal_risk_eda_figures/EDA.png src/maternal_risk_eda_figures/box_plot.png src/maternal_risk_eda_figures/class_distribution src/maternal_risk_eda_figures/density_plot.png src/maternal_risk_eda_figures/output_32_0.png :src/eda_script.py data/raw/maternal_risk.csv
	python src/eda_script.py --data_location='data/raw/maternal_risk.csv' --output_location='src/maternal_risk_eda_figures/'


# create the figures from model building 
src/maternal_risk_model_figures/hyperparam_plot.png src/maternal_risk_model_figures/model_comparison_table.csv : src/fit_maternal_risk_predict_model.py data/processed/train_df.csv data/processed/test_df.csv 
	python src/fit_maternal_risk_predict_model.py --train_df_path='data/processed/train_df.csv' --test_df_path='data/processed/test_df.csv' --output_dir_path='src/maternal_risk_model_figures/'



# test model 


# create the final report 
doc/final_report.md : doc/final_report.Rmd data/processed/test_df.csv src/maternal_risk_eda_figures/box_plot.png data/processed/test_df.csv src/maternal_risk_model_figures/hyperparam_plot.png
	Rscript -e "rmarkdown::render('doc/final_report.Rmd')"


clean :
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv
	rm -f src/maternal_risk_eda_figures/*.png
	rm -f src/maternal_risk_model_figures/*.png
	rm -f src/maternal_risk_model_figures/*.csv