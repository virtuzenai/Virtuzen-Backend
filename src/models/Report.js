const mongoose = require('mongoose');

const ReportSchema = new mongoose.Schema({
  timeframe: { type: String, required: true },
  performance: {
    labels: [String],
    successRates: [Number],
    executionTimes: [Number],
  },
  history: [{
    workflow: String,
    status: String,
    timestamp: Date,
  }],
  stats: {
    totalWorkflows: Number,
    successRate: Number,
    avgExecutionTime: Number,
    failedRuns: Number,
  },
  createdAt: { type: Date, default: Date.now },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
});

module.exports = mongoose.model('Report', ReportSchema);
