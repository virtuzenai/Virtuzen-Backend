const mongoose = require('mongoose');

const AutomationSchema = new mongoose.Schema({
  name: { type: String, required: true },
  type: { type: String, enum: ['email', 'task', 'data', 'custom'], required: true },
  trigger: { type: String, enum: ['immediate', 'scheduled', 'event'], required: true },
  scheduleTime: { type: String, required: function() { return this.trigger === 'scheduled'; } },
  status: { type: String, enum: ['running', 'scheduled', 'paused', 'completed', 'error'], required: true },
  createdAt: { type: String, required: true },
  progress: { type: Number, default: 0 },
});

module.exports = mongoose.model('Automation', AutomationSchema);
