# Start with an official Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project folder into the container
COPY . /app

# Install necessary Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Shiny (Shiny for Python runs on 8000)
EXPOSE 8000

# Run the Shiny app
CMD ["python", "-m", "shiny", "run", "scripts/app.py", "--host", "0.0.0.0", "--port", "8000"]
