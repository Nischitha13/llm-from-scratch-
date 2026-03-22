# Use official Python 3.10 on Linux x86_64
# --platform=linux/amd64 forces x86_64 emulation — this is the key fix that
# allows grain, array-record, and jaxlib to install (they have no Apple Silicon wheels)
FROM --platform=linux/amd64 python:3.10-slim

# Don't buffer Python output — logs appear in real time in the terminal
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
# All subsequent commands run from /app
WORKDIR /app

# Copy requirements.txt FIRST (before the rest of the project)
# Docker caches each step — if requirements.txt hasn't changed,
# Docker skips re-installing all packages on the next build (much faster)
COPY requirements.txt .

# Install all Python packages
# --no-cache-dir avoids storing the download cache, keeping the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into /app
COPY . .

# Tell Docker this container listens on port 8888 (Jupyter's default)
EXPOSE 8888

# Command that runs when the container starts:
#   --ip=0.0.0.0       listen on all interfaces (required so your Mac can connect)
#   --port=8888        use port 8888
#   --no-browser       don't try to open a browser (there's no desktop in a container)
#   --allow-root       allow running as root inside the container
#   --NotebookApp.token=''  disable the login token for convenience
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
