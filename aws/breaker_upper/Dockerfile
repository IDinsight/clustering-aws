# Use miniconda as base image
FROM continuumio/miniconda3:23.10.0-1

# Copy function code
COPY breaker_upper.py .

RUN conda install pandas -y
RUN pip install --no-cache-dir awswrangler==3.5.2 --target .

# Add Lambda Runtime Interface Client
RUN pip install awslambdaric --target .

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/opt/conda/bin/python", "-m", "awslambdaric" ]

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "breaker_upper.handler" ]
