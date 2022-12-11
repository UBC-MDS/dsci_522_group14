# Docker file for the Maternal Health Predictor project
# December 6, 2022

# use continuumio/miniconda3 as the base image
FROM continuumio/miniconda3:4.12.0

RUN apt update && apt install -y make

### PYTHON
# override/install python 3.10 
RUN conda install -y python=3.10

