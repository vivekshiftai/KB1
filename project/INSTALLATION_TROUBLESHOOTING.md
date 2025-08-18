# Installation Troubleshooting Guide

This guide helps resolve common installation issues with the PDF Intelligence Platform.

## 🚨 Common Installation Errors

### 1. **PyTorch Installation Issues**

**Error:** `ERROR: Could not find a version that satisfies the requirement torch`

**Solution:**
```bash
# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. **ChromaDB Installation Issues**

**Error:** `ERROR: Failed building wheel for hnswlib`

**Solution:**
```bash
# Install system dependencies first (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Then install ChromaDB
pip install chromadb
```

### 3. **PyMuPDF Installation Issues**

**Error:** `ERROR: Failed building wheel for PyMuPDF`

**Solution:**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libmupdf-dev

# Then install PyMuPDF
pip install PyMuPDF
```

### 4. **Sentence Transformers Issues**

**Error:** `ERROR: Could not find a version that satisfies the requirement sentence-transformers`

**Solution:**
```bash
# Update pip first
pip install --upgrade pip

# Install with specific version
pip install sentence-transformers==2.2.2
```

### 5. **Magic-PDF Installation Issues**

**Error:** `ERROR: Failed building wheel for magic-pdf`

**Solution:**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libmagic1

# For Windows
pip install python-magic-bin

# Then install magic-pdf
pip install magic-pdf
```

## 🔧 Platform-Specific Solutions

### **Windows**

1. **Install Visual Studio Build Tools:**
   ```bash
   # Download and install from Microsoft
   # https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```

2. **Use Windows-specific packages:**
   ```bash
   pip install python-magic-bin
   ```

### **macOS**

1. **Install Homebrew dependencies:**
   ```bash
   brew install libmagic
   brew install mupdf
   ```

2. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

### **Linux (Ubuntu/Debian)**

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential python3-dev libmagic1 libmupdf-dev
   ```

2. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Alternative Installation Methods

### **Using Conda**

```bash
# Create conda environment
conda create -n pdf-intelligence python=3.9
conda activate pdf-intelligence

# Install packages
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install -r requirements.txt
```

### **Using Docker**

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libmagic1 \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Run application
CMD ["python", "main.py"]
```

## 🔍 Step-by-Step Installation

### **1. Check Python Version**
```bash
python --version
# Should be Python 3.8 or higher
```

### **2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Upgrade Pip**
```bash
pip install --upgrade pip
```

### **4. Install System Dependencies (Linux/macOS)**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential python3-dev libmagic1 libmupdf-dev

# macOS
brew install libmagic mupdf
```

### **5. Install Python Dependencies**
```bash
# Try minimal requirements first
pip install -r requirements-minimal.txt

# If successful, install full requirements
pip install -r requirements.txt
```

### **6. Verify Installation**
```bash
python -c "import fastapi, uvicorn, chromadb, sentence_transformers, openai, fitz; print('All packages installed successfully!')"
```

## 🐛 Common Runtime Issues

### **1. OpenAI API Key Not Set**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### **2. Port Already in Use**
```bash
# Check what's using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/macOS

# Use different port
uvicorn main:app --port 8001
```

### **3. Memory Issues**
```bash
# Reduce batch size in .env
echo "MAX_CHUNKS_PER_BATCH=10" >> .env
```

## 📞 Getting Help

If you're still experiencing issues:

1. **Check the logs:**
   ```bash
   python main.py 2>&1 | tee app.log
   ```

2. **Verify your environment:**
   ```bash
   python -c "import sys; print(sys.version)"
   pip list
   ```

3. **Try minimal installation:**
   ```bash
   pip install -r requirements-minimal.txt
   ```

4. **Create an issue** with:
   - Your operating system
   - Python version
   - Full error message
   - Steps to reproduce
