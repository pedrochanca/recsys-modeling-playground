# Use the official Python image as a base.
FROM python:3.12

# Set the working directory in the container.
WORKDIR /app

# Install system dependencies including Java
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    curl \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
RUN pip install --upgrade pip

# Copy the requirements.txt file into the container's working directory
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Set default command.
CMD ["python3"]