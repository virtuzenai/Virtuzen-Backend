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