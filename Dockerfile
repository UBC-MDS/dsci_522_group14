# test
FROM continuumio/miniconda3:4.12.0

### Python
RUN conda install -y python=3.10
RUN conda config --append channels conda-forge

### Conda
RUN conda install -y\
    ipykernel=6.17.1 \
    scikit-learn>=1.1.3 \
    altair=4.2.0 \
    altair_saver=0.1.0 \
    matplotlib=3.6.2\ 
    pandas=1.4.4 \
    pandoc>=1.12.3
    
RUN conda install -y -c jmcmurray os

### Pip 
RUN apt-get update && apt-get install -y pip

RUN pip install \
    docopt==0.6.2 \
    joblib==1.1.0 \
    selenium==4.2.0 \
    vl-convert-python==0.5.0 \
    shutup==0.2.0

RUN pip install numpy==1.23.5 
RUN pip install regex==2022.10.31
RUN pip install requests==2.22.0
RUN pip install graphviz==0.20.1
RUN pip install nbconvert==7.2.5
RUN pip install scipy==1.9.3

### R
RUN apt-get install r-base r-base-dev -y

RUN apt-get update && apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev


RUN R -q -e 'install.packages("tidyverse")'
RUN R -q -e 'install.packages("rmarkdown")'
RUN R -e "install.packages('knitr')"

### Make
RUN apt update && apt install -y make
