const express = require('express');
const router = express.Router();
const auth = require('./authMiddleware');
const validate = require('./validate');
const Joi = require('joi');
const Task = require('./Task');

const taskSchema = Joi.object({
  name: Joi.string().min(3).max(50).required(),
  trigger: Joi.string().valid('scheduled', 'event').required(),
  scheduleTime: Joi.date().when('trigger', { is: 'scheduled', then: Joi.required() }),
  event: Joi.string().when('trigger', { is: 'event', then: Joi.required() }),
  description: Joi.string().max(500),
});

router.get('/', auth, async (req, res, next) => {
  try {
    const tasks = await Task.find({ userId: req.userId }).sort({ createdAt: -1 });
    res.json(tasks);
  } catch (error) {
    next(error);
  }
});

router.post('/', auth, validate(taskSchema), async (req, res, next) => {
  try {
    const task = new Task({ ...req.body, userId: req.userId });
    await task.save();
    res.status(201).json(task);
  } catch (error) {
    next(error);
  }
});

router.delete('/:id', auth, async (req, res, next) => {
  try {
    const task = await Task.findOneAndDelete({ _id: req.params.id, userId: req.userId });
    if (!task) throw new Error('Task not found');
    res.json({ message: 'Task deleted' });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
