// server.js
const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const app = express();
require('dotenv').config();

app.use(express.json());

// Initialize Gemini with two different API keys
const generalGenAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY_GENERAL);
const tutorGenAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY_TUTOR);

const generalModel = generalGenAI.getGenerativeModel({ model: "gemini-pro" });
const tutorModel = tutorGenAI.getGenerativeModel({ model: "gemini-pro" }); // Adjust model if different

// General Chat Endpoint (No Auth)
app.post('/api/chat/general', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: 'Message required' });

    const result = await generalModel.generateContent(message);
    res.json({ candidates: [{ content: { parts: [{ text: result.response.text() }] } }] });
  } catch (error) {
    console.error('General Chat Error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Tutor Endpoint (No Auth)
app.post('/api/chat/tutor', async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: 'Message required' });

    const tutorPrompt = `You are Virtuzen Tutor, an expert educational AI. Provide a concise, step-by-step explanation for: ${message}`;
    const result = await tutorModel.generateContent(tutorPrompt);
    res.json({ candidates: [{ content: { parts: [{ text: result.response.text() }] } }] });
  } catch (error) {
    console.error('Tutor Error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Serve static files from the 'public' directory
app.use(express.static('public'));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));