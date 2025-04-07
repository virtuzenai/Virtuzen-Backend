const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const UserSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true, trim: true },
  email: { type: String, required: true, unique: true, trim: true },
  password: { type: String, required: true },
  settings: {
    pushNotifications: { type: Boolean, default: false },
    emailAlerts: { type: Boolean, default: false },
    inAppAlerts: { type: Boolean, default: true },
    defaultTrigger: { type: String, enum: ['manual', 'scheduled', 'event'], default: 'manual' },
    executionTimeout: { type: Number, default: 30, min: 1 }
  },
  createdAt: { type: Date, default: Date.now }
});

UserSchema.pre('save', async function(next) {
  if (this.isModified('password')) {
    this.password = await bcrypt.hash(this.password, 10);
  }
  next();
});

UserSchema.methods.comparePassword = async function(password) {
  return await bcrypt.compare(password, this.password);
};

module.exports = mongoose.model('User', UserSchema);
