require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const http = require('http');
const connectDB = require('./db');
const logger = require('./logger');
const errorHandler = require('./errorHandler');
const limiter = require('./rateLimiter');
const { initSocket } = require('./socket');
const { scheduleWorkflows } = require('./workflowScheduler');

const app = express();
const server = http.createServer(app);

app.use(cors({ origin: '*' }));
app.use(helmet());
app.use(express.json());
app.use(limiter);

app.use('/auth', require('./auth'));
app.use('/dashboard', require('./dashboard'));
app.use('/settings', require('./settings'));
app.use('/workflows', require('./workflows'));
app.use('/reports', require('./reports'));
app.use('/tasks', require('./tasks'));

initSocket(server);

app.use(errorHandler);

const PORT = process.env.PORT || 3000;
const startServer = async () => {
  try {
    await connectDB();
    server.listen(PORT, () => {
      logger.info(`Server running on port ${PORT}`);
      scheduleWorkflows();
    });
  } catch (error) {
    logger.error('Server startup failed:', error);
    process.exit(1);
  }
};

startServer();
