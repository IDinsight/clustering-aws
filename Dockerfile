# Use miniconda as base image
FROM continuumio/miniconda3:23.10.0-1

# install geopandas through conda to avoid gdal issues
RUN conda install geopandas==0.14.2 -y

# copy clustering module files
COPY clustering /clustering
COPY requirements_lambda.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements_lambda.txt --target .

# Copy function code
COPY aws/app.py .

# Add Lambda Runtime Interface Client
RUN pip install awslambdaric --target .

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/opt/conda/bin/python", "-m", "awslambdaric" ]

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]
