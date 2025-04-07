const express = require('express');
const router = express.Router();
const auth = require('../middleware/auth');
const Workflow = require('../models/Workflow');
const Report = require('../models/Report');

router.get('/', auth, async (req, res, next) => {
  try {
    const userId = req.userId;
    const activeWorkflows = await Workflow.find({ userId, status: 'running' }).lean();
    const recentReports = await Report.find({ userId }).sort({ timestamp: -1 }).limit(5).lean();
    const stats = {
      totalWorkflows: await Workflow.countDocuments({ userId }),
      successRate: (await Report.aggregate([
        { $match: { userId } },
        { $group: { _id: null, success: { $sum: { $cond: [{ $eq: ['$status', 'completed'] }, 1, 0] }, total: { $sum: 1 } } },
        { $project: { rate: { $multiply: [{ $divide: ['$success', '$total'] }, 100] } } }
      ]))[0]?.rate || 0,
    };
    res.json({ activeWorkflows, recentReports, stats });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
