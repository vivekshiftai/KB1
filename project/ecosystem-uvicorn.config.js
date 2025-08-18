module.exports = {
  apps: [{
    name: 'pdf-intelligence-platform-uvicorn',
    script: 'uvicorn',
    args: 'main:app --host 0.0.0.0 --port 8000',
    cwd: '/home/ialuser/KB1/project',
    interpreter: '',
    env: {
      NODE_ENV: 'production',
      PYTHONPATH: '/home/ialuser/KB1/project'
    },
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    log_file: './logs/combined.log',
    out_file: './logs/out.log',
    error_file: './logs/error.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true
  }]
};
