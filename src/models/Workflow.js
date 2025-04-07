const mongoose = require('mongoose');

const ActionSchema = new mongoose.Schema({
  type: { type: String, required: true },
  details: { type: String, required: true },
});

const WorkflowSchema = new mongoose.Schema({
  name: { type: String, required: true },
  description: { type: String },
  trigger: { type: String, required: true },
  scheduleTime: { type: Date },
  actions: [ActionSchema],
  status: { type: String, default: 'pending' },
  lastUsed: { type: Date, default: Date.now },
  createdAt: { type: Date, default: Date.now },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
});

module.exports = mongoose.model('Workflow', WorkflowSchema);
