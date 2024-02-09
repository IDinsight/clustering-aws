# Use the official Python runtime as the base image
# FROM public.ecr.aws/lambda/python:3.11

# use miniconda as base image #anaconda-pkg-build
FROM continuumio/miniconda3:23.10.0-1

# install geopandas through conda to avoid gdal issues
RUN conda install geopandas==0.14.2 -y

# copy clustering module files
COPY clustering ${LAMBDA_TASK_ROOT}/clustering
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code - # REMOVE LOCAL
COPY aws/app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ] 
