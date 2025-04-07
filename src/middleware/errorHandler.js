const logger = require('../logger');

const errorHandler = (error, req, res, next) => {
  logger.error(`Error: ${error.message}`);
  res.status(400).json({ message: error.message || 'Something went wrong' });
};

module.exports = errorHandler;
