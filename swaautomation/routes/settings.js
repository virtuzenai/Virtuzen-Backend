const express = require('express');
const router = express.Router();
const auth = require('../middleware/auth');
const User = require('../models/User');
const validate = require('../middleware/validate');
const Joi = require('joi');

const settingsSchema = Joi.object({
  pushNotifications: Joi.boolean(),
  emailAlerts: Joi.boolean(),
  inAppAlerts: Joi.boolean(),
  defaultTrigger: Joi.string().valid('manual', 'scheduled', 'event'),
  executionTimeout: Joi.number().min(1),
});

router.get('/', auth, async (req, res, next) => {
  try {
    const user = await User.findById(req.userId).select('-password');
    res.json(user.settings);
  } catch (error) {
    next(error);
  }
});

router.put('/', auth, validate(settingsSchema), async (req, res, next) => {
  try {
    const user = await User.findById(req.userId);
    user.settings = { ...user.settings, ...req.body };
    await user.save();
    res.json({ message: 'Settings updated' });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
