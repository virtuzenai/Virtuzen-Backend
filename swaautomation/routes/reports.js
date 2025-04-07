const express = require('express');
const router = express.Router();
const auth = require('../middleware/auth');
const Report = require('../models/Report');

router.get('/', auth, async (req, res, next) => {
  try {
    const { timeframe = 'weekly' } = req.query;
    const dateRange = { daily: 1, weekly: 7, monthly: 30 }[timeframe] || 7;
    const startDate = new Date(Date.now() - dateRange * 24 * 60 * 60 * 1000);

    const reports = await Report.find({ userId: req.userId, timestamp: { $gte: startDate } })
      .populate('workflowId', 'name')
      .sort({ timestamp: -1 });

    const performance = await Report.aggregate([
      { $match: { userId: req.userId, timestamp: { $gte: startDate } } },
      { $group: {
        _id: { $dateToString: { format: '%Y-%m-%d', date: '$timestamp' } },
        successRates: { $avg: { $cond: [{ $eq: ['$status', 'completed'] }, 100, 0] } },
        executionTimes: { $avg: '$executionTime' },
      } },
      { $sort: { _id: 1 } },
    ]);

    const stats = {
      totalRuns: reports.length,
      successRate: performance.reduce((acc, p) => acc + p.successRates, 0) / performance.length || 0,
      avgExecutionTime: performance.reduce((acc, p) => acc + p.executionTimes, 0) / performance.length || 0,
    };

    res.json({ reports, performance, stats });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
