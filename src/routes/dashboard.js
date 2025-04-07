const express = require('express');
const router = express.Router();
const automationController = require('../controllers/automationController');

router.post('/start-automation', automationController.startAutomation);

module.exports = router;
