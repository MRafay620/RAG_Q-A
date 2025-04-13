# Use Ubuntu 24.04 as the base image
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Create a virtual environment
RUN python3 -m venv /app/venv && \
    chmod +x /app/venv/bin/activate && \
    # Install pip in the virtual environment
    /app/venv/bin/python -m pip install --upgrade pip && \
    # Install the dependencies in the virtual environment
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

WORKDIR /app/app

# Specify the default command to keep the container running indefinitely
CMD ["streamlit", "run", "app.py"]