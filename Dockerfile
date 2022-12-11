# Docker file for the Maternal Health Predictor project
# December 6, 2022

# use continuumio/miniconda3 as the base image
FROM continuumio/miniconda3:4.12.0

RUN apt update && apt install -y make

### PYTHON
# override/install python 3.10 
RUN conda install -y python=3.10

# install python packages with conda
#RUN conda install -y docopt=0.6.2
RUN conda install -y pandas=1.5.1
#RUN conda install -y pandoc
RUN conda install -y -c jmcmurray os
RUN conda install -y -c conda-forge altair_saver
RUN conda install -c conda-forge -y pandoc

# install python packages with pip 
RUN pip install numpy==1.23.5 
RUN pip install regex==2022.10.31
RUN pip install altair==4.2.0
RUN pip install requests==2.22.0
RUN pip install graphviz==0.20.1
RUN pip install nbconvert==7.2.5
RUN pip install scikit-learn==1.2.0
RUN pip install scipy==1.9.3
RUN pip install docopt-ng==0.8.*
RUN python -m pip install vl-convert-python==0.4.0

### R 
RUN apt-get install r-base r-base-dev -y

RUN Rscript -e "install.packages('rmarkdown')"
RUN Rscript -e "install.packages('knitr')"
# Install non R tidyverse dependencies
RUN apt-get update && apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev
# R Installs 
RUN R -q -e 'install.packages("tidyverse")'
RUN R -q -e 'install.packages("rmarkdown")'
