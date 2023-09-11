# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Run FastAPI with uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]
