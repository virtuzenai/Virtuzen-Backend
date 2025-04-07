const cron = require('node-cron');
const Workflow = require('./Workflow'); // Flat structure
const Report = require('./Report');     // Flat structure
const logger = require('./logger');
const { emitWorkflowUpdate } = require('./socket');

const scheduleWorkflows = async () => {
  const workflows = await Workflow.find({ trigger: 'scheduled', status: { $ne: 'running' } });
  workflows.forEach(workflow => {
    const scheduleTime = new Date(workflow.scheduleTime);
    const cronExpression = `${scheduleTime.getMinutes()} ${scheduleTime.getHours()} * * *`;

    cron.schedule(cronExpression, async () => {
      try {
        workflow.status = 'running';
        await workflow.save();
        emitWorkflowUpdate(workflow);

        await new Promise(resolve => setTimeout(resolve, 5000));
        workflow.status = 'completed';
        workflow.lastUsed = new Date();
        await workflow.save();

        await new Report({
          workflowId: workflow._id,
          status: 'completed',
          executionTime: 5,
          userId: workflow.userId,
        }).save();

        emitWorkflowUpdate(workflow);
        logger.info(`Workflow ${workflow.name} executed successfully`);
      } catch (error) {
        workflow.status = 'failed';
        await workflow.save();
        logger.error(`Workflow ${workflow.name} failed:`, error);
        emitWorkflowUpdate(workflow);
      }
    });
  });
};

module.exports = { scheduleWorkflows };
