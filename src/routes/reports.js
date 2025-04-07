const express = require('express');
const router = express.Router();
const Report = require('../models/Report');
const Workflow = require('../models/Workflow');
const logger = require('../logger');

// Generate and get report
router.get('/', async (req, res, next) => {
  try {
    const timeframe = req.query.timeframe || 'weekly';
    let report = await Report.findOne({ timeframe, userId: req.user.id }).sort({ createdAt: -1 });

    if (!report) {
      const workflows = await Workflow.find({ userId: req.user.id });
      const history = workflows.map(w => ({
        workflow: w.name,
        status: w.status,
        timestamp: w.lastUsed,
      }));

      const stats = {
        totalWorkflows: workflows.length,
        successRate: workflows.length ? (workflows.filter(w => w.status === 'completed').length / workflows.length) * 100 : 0,
        avgExecutionTime: 6.5, // Simulate for now
        failedRuns: workflows.filter(w => w.status === 'failed').length,
      };

      report = new Report({
        timeframe,
        performance: {
          labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
          successRates: [95, 88, 92, 85, 90, 87, 93],
          executionTimes: [5, 7, 4, 6, 5, 8, 6],
        },
        history,
        stats,
        userId: req.user.id,
      });

      await report.save();
    }

    res.json(report);
  } catch (error) {
    next(error);
  }
});

module.exports = router;
