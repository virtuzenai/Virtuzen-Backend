require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const connectDB = require('./src/config/db');
const errorHandler = require('./src/middleware/errorHandler');
const authMiddleware = require('./src/middleware/auth');
const dashboardRoutes = require('./src/routes/dashboard');
const workflowRoutes = require('./src/routes/workflows');
const taskRoutes = require('./src/routes/tasks');
const reportRoutes = require('./src/routes/reports');
const settingsRoutes = require('./src/routes/settings');
const logger = require('./src/logger');
const path = require('path');

const app = express();

// Middleware
app.use(helmet());
app.use(cors({ origin: process.env.FRONTEND_URL || '*' }));
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public'))); // Serve static files

// Rate Limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
});
app.use(limiter);

// Connect to MongoDB
connectDB();

// Routes
app.use('/dashboard', authMiddleware, dashboardRoutes);
app.use('/workflows', authMiddleware, workflowRoutes);
app.use('/tasks', authMiddleware, taskRoutes);
app.use('/reports', authMiddleware, reportRoutes);
app.use('/settings', authMiddleware, settingsRoutes);

// Serve the frontend
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Error Handling
app.use(errorHandler);

// Start Server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  logger.info(`Server running on port ${PORT}`);
});
=======
const express = require("express");
const axios = require("axios");
const cors = require("cors");
require("dotenv").config();

const app = express();
app.use(express.json());
app.use(cors());

const API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent";
const API_KEY = process.env.GEMINI_API_KEY;

app.post("/api/chat", async (req, res) => {
    try {
        const userMessage = req.body.message;
        const type = req.body.type || "general"; // Default to "general"

        // Customizing prompt for different types
        let prompt;
        if (type === "tutor") {
            prompt = `You are a tutor. Explain in detail: ${userMessage}`;
        } else {
            prompt = userMessage;
        }

        const response = await axios.post(
            `${API_URL}?key=${API_KEY}`,
            {
                contents: [{ parts: [{ text: prompt }] }]
            },
            { headers: { "Content-Type": "application/json" } }
        );

        res.json({ response: response.data });
    } catch (error) {
        console.error("Error:", error.response ? error.response.data : error.message);
        res.status(500).json({ error: "AI response failed" });
    }
});

app.listen(3000, () => console.log("Server running on port 3000"));

