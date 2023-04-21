# Use the official Python base image with the desired version
FROM python:3.7.4

# Set the working directory inside the container
WORKDIR /app

# Clone the GitHub repository
RUN git clone https://github.com/shahcompbio/scdna_replication_tools.git

# Change to the cloned repository's directory
WORKDIR /app/scdna_replication_tools

# # Create a virtual environment
# RUN python -m venv venv/

# # Activate the virtual environment
# RUN /bin/bash -c "source venv/bin/activate"

# Install dependencies
RUN pip install numpy==1.21.4 cython==0.29.22
RUN pip install -r requirements4.txt

# Install the package in development mode
RUN python setup.py develop

# Expose any required ports
# EXPOSE <port_number>

# Run any required commands or scripts to start the application
# CMD <command>