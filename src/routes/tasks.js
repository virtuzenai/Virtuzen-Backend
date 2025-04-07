const express = require('express');
const router = express.Router();
const Task = require('../models/Task');
const logger = require('../logger');

// Get all tasks
router.get('/', async (req, res, next) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    const tasks = await Task.find({ userId: req.user.id })
      .skip(skip)
      .limit(limit)
      .sort({ createdAt: -1 });

    const total = await Task.countDocuments({ userId: req.user.id });
    res.json({
      tasks,
      pagination: {
        page,
        limit,
        totalPages: Math.ceil(total / limit),
        total,
      },
    });
  } catch (error) {
    next(error);
  }
});

// Create a task
router.post('/', async (req, res, next) => {
  try {
    const { name, trigger, scheduleTime, event, description } = req.body;
    if (!name || !trigger) {
      return res.status(400).json({ message: 'Missing required fields' });
    }

    const task = new Task({
      name,
      trigger,
      scheduleTime: trigger === 'scheduled' ? scheduleTime : undefined,
      event: trigger === 'event' ? event : undefined,
      description,
      userId: req.user.id,
    });

    await task.save();
    res.status(201).json(task);
  } catch (error) {
    next(error);
  }
});

module.exports = router;
