# Docker file for the Maternal Health Predictor project
# December 6, 2022

# use rocker/tidyverse as the base image 
#FROM rocker/tidyverse 
#**FROM debian:stable

# install R packages
#RUN apt-get update
#RUN apt-get install r-base r-base-dev -y
#RUN Rscript -e "install.packages('knitr')"
#RUN Rscript -e "install.packages('tidyverse')"
#RUN Rscript -e "install.packages('kableExtra')" 

# install anaconda & put it in the PATH
# attribution: Tiffany Timbers (https://github.com/UBC-DSCI/introduction-to-datascience/blob/b0f86fc4d6172cd043a0eb831b5d5a8743f29c81/Dockerfile#L19)
#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
#    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#    rm ~/anaconda.sh 
#ENV PATH /opt/conda/bin:$PATH

# install Python packages 
#RUN /opt/conda/bin/conda install -y -c anaconda docopt
#RUN /opt/conda/bin/conda install -y anaconda graphviz

#######################################################

# use continuumio/miniconda3 as the base image
FROM continuumio/miniconda3:4.12.0

RUN apt update && apt install -y make

### PYTHON
# override/install python 3.10 
RUN conda install -y python=3.10

# install other python packages with conda
#RUN conda install -y docopt=0.6.2
RUN conda install -y pandas=1.5.1
RUN conda install -y pandoc=2
RUN conda install -y -c jmcmurray os

# install other python packages with pip 
RUN pip install numpy==1.23.5 
RUN pip install regex==2022.10.31
RUN pip install altair==4.2.0
RUN pip install requests==2.22.0
RUN pip install graphviz==0.20.1
RUN pip install nbconvert==7.2.5
RUN pip install scikit-learn==1.2.0
RUN pip install scipy==1.9.3

# other 
RUN conda install -y -c conda-forge altair_saver
#=0.5.0
RUN python -m pip install vl-convert-python==0.4.0

### R 
# install R
RUN apt-get install r-base r-base-dev -y

# install other R packages 
RUN Rscript -e "install.packages('rmarkdown')"
RUN Rscript -e "install.packages('knitr')"


#RUN Rscript -e "install.packages('libcurl4-openssl-dev')"
#RUN Rscript -e "install.packages('libssl-dev')"
#RUN Rscript -e "install.packages('libxml2-dev')"
#RUN Rscript -e "install.packages('xml2')"
#RUN Rscript -e "install.packages('rvest')"

#RUN Rscript -e "install.packages('tidyverse')"
#RUN Rscript -e "install.packages('tidyr')"

#RUN Rscript -e "install.packages('devtools')"
#RUN Rscript -e "devtools::install_github('tidyverse/tidyverse')"
#RUN Rscript -e "install.packages('kableExtra')" 
#RUN Rscript -e "install.packages('readr')" 


RUN pip install docopt-ng==0.8.*
RUN pip install dataframe-image==0.1.3
#RUN apt install chromium-chromedriver
RUN pip install dataframe-image==0.1.3


###
# Install R
RUN apt-get install r-base r-base-dev -y

# Install non R tidyverse dependencies
RUN apt-get update && apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev

# R Installs 
RUN R -q -e 'install.packages("tidyverse")'
RUN R -q -e 'install.packages("rmarkdown")'

# Install Make
RUN apt update && apt install -y make
