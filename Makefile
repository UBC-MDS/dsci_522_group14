# Maternal Health Risk Predictor Makefile Pipeline
# Author: Lennon Au-Yeung, Chenyang Wang, Shirley Zhang
# Date: 2022-11-29

# This driver script completes data preprocessing, exploratory data analysis, 
# and model building for a classification model which predicts maternal health 
# risk based on various health metrics. It creates figures and tables from the 
# analyses and compiles it all in a final report. It takes no arguments. 

# Example usage: 
# make all 
# make src/maternal_risk_eda_figures/EDA.png
# make clean 



all : doc/final_report.md

# download data from url 
data/raw/maternal_risk.csv : src/download_data.py
	python src/download_data.py --out_type='csv' --url='https://archive.ics.uci.edu/ml/machine-learning-databases/00639/Maternal%20Health%20Risk%20Data%20Set.csv' --out_file='data/raw/maternal_risk.csv'


# preprocess data by splitting into train and test set 
data/processed/test_df.csv data/processed/test_df_binary.csv data/processed/train_df.csv data/processed/train_df_binary.csv : data/raw/maternal_risk.csv src/pre_processing.py 
	python src/pre_processing.py --data_location='data/raw/maternal_risk.csv' --output_location='data/processed/'


# create figures from exploratory data analysis 
src/maternal_risk_eda_figures/EDA.png src/maternal_risk_eda_figures/box_plot.png src/maternal_risk_eda_figures/class_distribution.png src/maternal_risk_eda_figures/density_plot.png src/maternal_risk_eda_figures/corr_bp_plot.png : data/raw/maternal_risk.csv src/eda_script.py 
	python src/eda_script.py --data_location='data/raw/maternal_risk.csv' --output_location='src/maternal_risk_eda_figures/'


# create graphs and figures from model building and hyperparameter optimization
results/hyperparam_plot.png results/model_comparison_table.csv : data/processed/train_df.csv data/processed/test_df.csv src/fit_maternal_risk_predict_model.py 
	python src/fit_maternal_risk_predict_model.py --train_df_path='data/processed/train_df.csv' --test_df_path='data/processed/test_df.csv' --output_dir_path='results/'


# score model on test data and create confusion matrix
results/test_score.csv results/testdata_confusion_matrix.csv : results/bestmodel.pkl data/processed/test_df.csv src/predict_model_on_test.py
	python src/predict_model_on_test.py --bestmodel_path='results/bestmodel.pkl' --test_df_path='data/processed/test_df.csv' --output_dir_path='results/'


# create the final report 
doc/final_report.md : data/processed/test_df.csv src/maternal_risk_eda_figures/box_plot.png data/processed/test_df.csv results/hyperparam_plot.png results/testdata_confusion_matrix.csv doc/final_report.Rmd
	Rscript -e "rmarkdown::render('doc/final_report.Rmd')"


clean :
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv
	rm -f src/maternal_risk_eda_figures/*.png
	rm -f results/*.png
	rm -f results/*.csv
	rm -f results/*.pkl