const mongoose = require('mongoose');

const WorkflowSchema = new mongoose.Schema({
  name: { type: String, required: true, trim: true },
  description: { type: String, trim: true },
  trigger: { type: String, enum: ['manual', 'scheduled', 'event'], required: true },
  scheduleTime: { type: Date, required: function() { return this.trigger === 'scheduled'; } },
  actions: [{
    type: { type: String, enum: ['sendEmail', 'updateDatabase', 'runScript'], required: true },
    details: { type: String, required: true }
  }],
  status: { type: String, enum: ['pending', 'running', 'completed', 'failed'], default: 'pending' },
  lastUsed: { type: Date, default: Date.now },
  createdAt: { type: Date, default: Date.now },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true }
});

module.exports = mongoose.model('Workflow', WorkflowSchema);
