const mongoose = require('mongoose');

const TaskSchema = new mongoose.Schema({
  name: { type: String, required: true, trim: true },
  trigger: { type: String, enum: ['scheduled', 'event'], required: true },
  scheduleTime: { type: Date, required: function() { return this.trigger === 'scheduled'; } },
  event: { type: String, required: function() { return this.trigger === 'event'; } },
  description: { type: String, trim: true },
  status: { type: String, enum: ['pending', 'completed', 'failed'], default: 'pending' },
  createdAt: { type: Date, default: Date.now },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true }
});

module.exports = mongoose.model('Task', TaskSchema);
