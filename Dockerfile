# Use the official Python runtime as the base image
# FROM public.ecr.aws/lambda/python:3.11

# use miniconda as base image
FROM continuumio/miniconda3


# Update the package lists and install GDAL
# RUN conda install gdal
RUN conda install geopandas==0.14.2 -y

# copy test data
COPY data ${LAMBDA_TASK_ROOT}/data

# copy clustering module files
COPY clustering ${LAMBDA_TASK_ROOT}/clustering
COPY pyproject.toml ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# install --no-cache-dir  --target ${LAMBDA_TASK_ROOT} 
RUN pip install --upgrade pip
RUN pip install .

# Copy function code - # REMOVE LOCAL
COPY aws/app_local.py ${LAMBDA_TASK_ROOT} 

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app_local.handler" ] 
