const Automation = require('../models/Automation');
const logger = require('../logger');
const cron = require('node-cron');

const startAutomation = async (req, res, next) => {
  try {
    const { name, type, trigger, scheduleTime, status, createdAt } = req.body;

    // Validation
    if (!name) throw new Error('Automation name is required');
    if (!['email', 'task', 'data', 'custom'].includes(type)) throw new Error('Invalid automation type');
    if (!['immediate', 'scheduled', 'event'].includes(trigger)) throw new Error('Invalid trigger');
    if (trigger === 'scheduled' && !scheduleTime) throw new Error('Schedule time is required for scheduled automations');

    // Create automation
    const automation = new Automation({
      name,
      type,
      trigger,
      scheduleTime,
      status,
      createdAt,
      progress: trigger === 'immediate' ? 0 : undefined,
    });

    // Save to database
    await automation.save();

    // Schedule task if needed
    if (trigger === 'scheduled') {
      const scheduleDate = new Date(scheduleTime);
      const cronExpression = `${scheduleDate.getMinutes()} ${scheduleDate.getHours()} ${scheduleDate.getDate()} ${scheduleDate.getMonth() + 1} *`;
      cron.schedule(cronExpression, async () => {
        logger.info(`Running scheduled automation: ${name}`);
        automation.status = 'running';
        automation.progress = 0;
        await automation.save();
        // Add actual automation logic here (e.g., send email, process data)
      });
    }

    // Simulate immediate automation (replace with real logic)
    if (trigger === 'immediate') {
      setTimeout(async () => {
        automation.progress = 100;
        automation.status = 'completed';
        await automation.save();
        logger.info(`Automation ${name} completed`);
      }, 5000); // Simulate 5-second task
    }

    res.status(200).json({ message: 'Automation started successfully!' });
  } catch (error) {
    logger.error('Error starting automation:', error);
    next(error);
  }
};

module.exports = { startAutomation };
