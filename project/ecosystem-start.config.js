module.exports = {
  apps: [{
    name: 'pdf-intelligence-platform',
    script: 'start.py',
    cwd: '/home/ialuser/KB1/project',
    interpreter: 'python3',
    env: {
      NODE_ENV: 'production',
      HOST: '0.0.0.0',
      PORT: '8000',
      RELOAD: 'false',
      LOG_LEVEL: 'info'
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
