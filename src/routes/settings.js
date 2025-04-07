const express = require('express');
const router = express.Router();
const User = require('../models/User');
const logger = require('../logger');

// Get user settings
router.get('/', async (req, res, next) => {
  try {
    const user = await User.findById(req.user.id);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    res.json(user.settings);
  } catch (error) {
    next(error);
  }
});

// Update user settings
router.put('/', async (req, res, next) => {
  try {
    const updates = req.body;
    const user = await User.findByIdAndUpdate(
      req.user.id,
      { settings: updates },
      { new: true }
    );

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    res.json(user.settings);
  } catch (error) {
    next(error);
  }
});

module.exports = router;
