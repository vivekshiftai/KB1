# PM2 Setup Guide for PDF Intelligence Platform

This guide explains how to properly configure and run the PDF Intelligence Platform using PM2.

## 🚨 **Current Issue**

The error you're seeing occurs because PM2 is trying to run the uvicorn script as a JavaScript file. This happens when PM2 is not properly configured for Python applications.

## 🔧 **Solution**

I've created several PM2 configuration files to fix this issue:

### **Option 1: Using Startup Script (Recommended)**

1. **Make the startup script executable:**
   ```bash
   chmod +x start.py
   chmod +x pm2-start.sh
   ```

2. **Start the application:**
   ```bash
   ./pm2-start.sh start
   ```

3. **Or use PM2 directly:**
   ```bash
   pm2 start ecosystem-start.config.js
   ```

### **Option 2: Using Python Command**

```bash
pm2 start ecosystem.config.js
```

### **Option 3: Using Uvicorn Command**

```bash
pm2 start ecosystem-uvicorn.config.js
```

## 📋 **PM2 Commands**

### **Start the Application**
```bash
# Using the management script
./pm2-start.sh start

# Or directly with PM2
pm2 start ecosystem-start.config.js
```

### **Stop the Application**
```bash
./pm2-start.sh stop
# or
pm2 stop pdf-intelligence-platform
```

### **Restart the Application**
```bash
./pm2-start.sh restart
# or
pm2 restart pdf-intelligence-platform
```

### **Check Status**
```bash
./pm2-start.sh status
# or
pm2 status
```

### **View Logs**
```bash
./pm2-start.sh logs
# or
pm2 logs pdf-intelligence-platform
```

### **Monitor in Real-time**
```bash
pm2 monit
```

## 🛠️ **Configuration Files**

### **ecosystem-start.config.js** (Recommended)
- Uses `start.py` script
- Properly configured for Python
- Includes environment variables
- Logging configuration

### **ecosystem.config.js**
- Uses `python main.py` command
- Alternative approach
- Good for simple setups

### **ecosystem-uvicorn.config.js**
- Uses `uvicorn` command directly
- More explicit uvicorn configuration
- Good for advanced users

## 🔍 **Troubleshooting**

### **If you still get syntax errors:**

1. **Check Python path:**
   ```bash
   which python3
   which uvicorn
   ```

2. **Verify virtual environment:**
   ```bash
   source venv/bin/activate
   which python
   ```

3. **Test the startup script manually:**
   ```bash
   python3 start.py
   ```

4. **Check PM2 logs:**
   ```bash
   pm2 logs pdf-intelligence-platform --lines 100
   ```

### **Common Issues:**

1. **Wrong Python interpreter:**
   - Make sure PM2 uses the correct Python interpreter
   - Update the `interpreter` field in config files

2. **Missing dependencies:**
   - Ensure all requirements are installed
   - Activate virtual environment if using one

3. **Permission issues:**
   - Make sure scripts are executable
   - Check file permissions

## 📊 **Monitoring**

### **PM2 Dashboard**
```bash
pm2 monit
```

### **Application Status**
```bash
pm2 status
```

### **Process Information**
```bash
pm2 show pdf-intelligence-platform
```

### **Log Management**
```bash
# View all logs
pm2 logs

# View specific app logs
pm2 logs pdf-intelligence-platform

# Follow logs in real-time
pm2 logs pdf-intelligence-platform --follow

# Clear logs
pm2 flush
```

## 🔄 **Auto-restart on Server Reboot**

To make PM2 start automatically on server reboot:

```bash
# Save current PM2 configuration
pm2 save

# Setup PM2 to start on boot
pm2 startup

# Follow the instructions provided by the startup command
```

## 📝 **Environment Variables**

The configuration files include these environment variables:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `RELOAD`: Auto-reload mode (default: false)
- `LOG_LEVEL`: Logging level (default: info)
- `NODE_ENV`: Environment (default: production)

## 🎯 **Quick Start**

1. **Navigate to project directory:**
   ```bash
   cd /home/ialuser/KB1/project
   ```

2. **Make scripts executable:**
   ```bash
   chmod +x start.py pm2-start.sh
   ```

3. **Start the application:**
   ```bash
   ./pm2-start.sh start
   ```

4. **Check status:**
   ```bash
   ./pm2-start.sh status
   ```

5. **Access the application:**
   - API: http://your-server-ip:8000
   - Docs: http://your-server-ip:8000/docs
   - Health: http://your-server-ip:8000/health

## ✅ **Verification**

After starting, verify the application is running:

```bash
# Check PM2 status
pm2 status

# Check if the port is listening
netstat -tlnp | grep :8000

# Test the health endpoint
curl http://localhost:8000/health
```

The application should now run properly without the syntax error!
