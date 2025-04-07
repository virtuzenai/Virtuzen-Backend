const express = require('express');
const router = express.Router();
const auth = require('./authMiddleware');
const validate = require('./validate');
const Joi = require('joi');
const Workflow = require('./Workflow');
const Report = require('./Report');
const { emitWorkflowUpdate } = require('./socket');

const workflowSchema = Joi.object({
  name: Joi.string().min(3).max(50).required(),
  description: Joi.string().max(500),
  trigger: Joi.string().valid('manual', 'scheduled', 'event').required(),
  scheduleTime: Joi.date().when('trigger', { is: 'scheduled', then: Joi.required() }),
  actions: Joi.array().items(
    Joi.object({
      type: Joi.string().valid('sendEmail', 'updateDatabase', 'runScript').required(),
      details: Joi.string().required(),
    })
  ).min(1).required(),
});

router.get('/', auth, async (req, res, next) => {
  try {
    const workflows = await Workflow.find({ userId: req.userId }).sort({ lastUsed: -1 });
    res.json(workflows);
  } catch (error) {
    next(error);
  }
});

router.post('/', auth, validate(workflowSchema), async (req, res, next) => {
  try {
    const workflow = new Workflow({ ...req.body, userId: req.userId });
    await workflow.save();
    res.status(201).json(workflow);
  } catch (error) {
    next(error);
  }
});

router.put('/:id', auth, validate(workflowSchema), async (req, res, next) => {
  try {
    const workflow = await Workflow.findOneAndUpdate(
      { _id: req.params.id, userId: req.userId },
      req.body,
      { new: true, runValidators: true }
    );
    if (!workflow) throw new Error('Workflow not found');
    res.json(workflow);
  } catch (error) {
    next(error);
  }
});

router.delete('/:id', auth, async (req, res, next) => {
  try {
    const workflow = await Workflow.findOneAndDelete({ _id: req.params.id, userId: req.userId });
    if (!workflow) throw new Error('Workflow not found');
    res.json({ message: 'Workflow deleted' });
  } catch (error) {
    next(error);
  }
});

router.post('/:id/run', auth, async (req, res, next) => {
  try {
    const workflow = await Workflow.findOne({ _id: req.params.id, userId: req.userId });
    if (!workflow) throw new Error('Workflow not found');
    if (workflow.status === 'running') throw new Error('Workflow already running');

    workflow.status = 'running';
    await workflow.save();
    emitWorkflowUpdate(workflow);

    setTimeout(async () => {
      workflow.status = 'completed';
      workflow.lastUsed = new Date();
      await workflow.save();
      await new Report({
        workflowId: workflow._id,
        status: 'completed',
        executionTime: 5,
        userId: req.userId,
      }).save();
      emitWorkflowUpdate(workflow);
    }, 5000);

    res.json({ message: 'Workflow started' });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
