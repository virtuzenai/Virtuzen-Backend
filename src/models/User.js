const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  settings: {
    notifications: {
      pushNotifications: { type: Boolean, default: false },
      emailAlerts: { type: Boolean, default: false },
      inAppAlerts: { type: Boolean, default: true },
    },
    workflowPrefs: {
      defaultTrigger: { type: String, default: 'manual' },
      executionTimeout: { type: Number, default: 30 },
    },
    aiCustomization: {
      responseStyle: { type: String, default: 'formal' },
      voice: { type: String, default: 'default' },
    },
    security: {
      twoFactorAuth: { type: Boolean, default: false },
      dataEncryption: { type: Boolean, default: true },
      permissions: { type: String, default: 'admin' },
    },
  },
  createdAt: { type: Date, default: Date.now },
});

module.exports = mongoose.model('User', UserSchema);
