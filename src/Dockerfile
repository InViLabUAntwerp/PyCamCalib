# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV and basic compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
# (This includes app.py, templates/, static/, and PyCamCalib/)
COPY . .

# Create the uploads folder just in case
RUN mkdir -p uploads

# Expose the port Gunicorn will run on
EXPOSE 8000

# Set environment variables
ENV MPLBACKEND=Agg
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the app using Gunicorn (as defined in your comment)
CMD ["gunicorn", "--chdir", "PyCamCalib/camera_calibration_flaskv2/", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8000", "app:app"]
#gunicorn --chdir PyCamCalib/camera_calibration_flask/ -w 1 --threads 4 -b 0.0.0.0:8000 app:app