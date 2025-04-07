const express = require('express');
const router = express.Router();
const Workflow = require('../models/Workflow');
const logger = require('../logger');

// Get all workflows
router.get('/', async (req, res, next) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    const workflows = await Workflow.find({ userId: req.user.id })
      .skip(skip)
      .limit(limit)
      .sort({ createdAt: -1 });

    const total = await Workflow.countDocuments({ userId: req.user.id });
    res.json({
      workflows,
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

// Create a workflow
router.post('/', async (req, res, next) => {
  try {
    const { name, description, trigger, scheduleTime, actions } = req.body;
    if (!name || !trigger || !actions || actions.length === 0) {
      return res.status(400).json({ message: 'Missing required fields' });
    }

    const workflow = new Workflow({
      name,
      description,
      trigger,
      scheduleTime: trigger === 'scheduled' ? scheduleTime : undefined,
      actions,
      userId: req.user.id,
    });

    await workflow.save();
    res.status(201).json(workflow);
  } catch (error) {
    next(error);
  }
});

// Update a workflow
router.put('/:id', async (req, res, next) => {
  try {
    const { id } = req.params;
    const updates = req.body;

    const workflow = await Workflow.findOneAndUpdate(
      { _id: id, userId: req.user.id },
      { ...updates, lastUsed: new Date() },
      { new: true }
    );

    if (!workflow) {
      return res.status(404).json({ message: 'Workflow not found' });
    }

    res.json(workflow);
  } catch (error) {
    next(error);
  }
});

// Delete a workflow
router.delete('/:id', async (req, res, next) => {
  try {
    const { id } = req.params;
    const workflow = await Workflow.findOneAndDelete({ _id: id, userId: req.user.id });
    if (!workflow) {
      return res.status(404).json({ message: 'Workflow not found' });
    }
    res.json({ message: 'Workflow deleted successfully' });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
