const mongoose = require('mongoose');

const ReportSchema = new mongoose.Schema({
  workflowId: { type: mongoose.Schema.Types.ObjectId, ref: 'Workflow', required: true },
  status: { type: String, enum: ['completed', 'failed', 'running'], required: true },
  executionTime: { type: Number, required: true }, // in seconds
  timestamp: { type: Date, default: Date.now },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true }
});

module.exports = mongoose.model('Report', ReportSchema);
