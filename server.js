require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const connectDB = require('./src/config/db');
const errorHandler = require('./src/middleware/errorHandler');
const dashboardRoutes = require('./src/routes/dashboard');
const logger = require('./src/logger');

const app = express();

// Middleware
app.use(helmet()); // Security headers
app.use(cors({ origin: process.env.FRONTEND_URL || '*' })); // Adjust for your frontend URL
app.use(express.json());

// Connect to MongoDB
connectDB();

// Routes
app.use('/dashboard', dashboardRoutes);

// Error handling
app.use(errorHandler);

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  logger.info(`Server running on port ${PORT}`);
});
