import imaplib
import smtplib
import email
from email.message import EmailMessage
from datetime import datetime, timedelta
import re
import time
import logging
import sys
import base64
import json
import os
from typing import Dict, List, Optional, Union
import threading
import queue
import hashlib
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Optional Gemini API and OCR
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    import pytesseract
    from PIL import Image
    from io import BytesIO
except ImportError:
    pytesseract = None
    Image = None
    BytesIO = None

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(funcName)s:%(lineno)d - %(process)d',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Email credentials (initially empty, will be set by user)
your_email = ""
your_app_password = ""
company_name = ""
agent_name = ""
forward_email = ""

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC4B45yRMZO2VVMzGYtLh-uW49Us0W-Ix8")
model = None
if genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini API initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {str(e)}")

# Database structure
HISTORY_FILE = "email_history.json"
email_history = {
    "contacts": {},
    "sent_emails": [],
    "threads": {},
    "scheduled_emails": [],
    "priority_queue": [],
    "spam_emails": [],
    "categories": {},
    "analytics": {
        "total_sent": 0,
        "avg_response_time": 0,
        "busiest_contacts": {},
        "response_trends": {}
    },
    "templates": {
        "welcome": "Dear {name}, welcome to {company}! How can I assist you today?",
        "thanks": "Dear {name}, thank you for your message. I’m here to help with any questions you have.",
        "follow_up": "Dear {name}, I’m following up on your request. Please let me know if there’s anything specific you need.",
        "urgent": "Dear {name}, I’m addressing your urgent request immediately. How can I assist you further?"
    },
    "rules": {},
    "tags": defaultdict(list)
}

# Load and save history functions
def load_history():
    global email_history
    try:
        if not os.path.exists(HISTORY_FILE) or os.stat(HISTORY_FILE).st_size == 0:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(email_history, f)
            logger.info("Created new empty history file.")
        with open(HISTORY_FILE, 'r') as f:
            loaded = json.load(f)
            for key in ["sent_emails", "scheduled_emails", "priority_queue", "spam_emails"]:
                for item in loaded.get(key, []):
                    if "timestamp" in item:
                        item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                    if "time" in item:
                        item["time"] = datetime.fromisoformat(item["time"])
                    if "last_contact" in item:
                        item["last_contact"] = datetime.fromisoformat(item["last_contact"])
            for contact in loaded["contacts"].values():
                contact["last_contact"] = datetime.fromisoformat(contact["last_contact"])
            email_history.update(loaded)
            logger.info("Loaded email history.")
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to load history: {str(e)}. Resetting.")
        email_history.clear()
        email_history.update({
            "contacts": {}, "sent_emails": [], "threads": {}, "scheduled_emails": [],
            "priority_queue": [], "spam_emails": [], "categories": {}, "analytics": {
                "total_sent": 0, "avg_response_time": 0, "busiest_contacts": {}, "response_trends": {}
            }, "templates": email_history["templates"], "rules": {}, "tags": defaultdict(list)
        })
        save_history()

def save_history():
    try:
        with open(HISTORY_FILE, 'w') as f:
            save_data = email_history.copy()
            for key in ["sent_emails", "scheduled_emails", "priority_queue", "spam_emails"]:
                for item in save_data[key]:
                    if "timestamp" in item:
                        item["timestamp"] = item["timestamp"].isoformat()
                    if "time" in item:
                        item["time"] = item["time"].isoformat()
                    if "last_contact" in item:
                        item["last_contact"] = item["last_contact"].isoformat()
            for contact in save_data["contacts"].values():
                contact["last_contact"] = contact["last_contact"].isoformat()
            save_data["tags"] = dict(save_data["tags"])
            json.dump(save_data, f)
            logger.debug("Saved history.")
    except Exception as e:
        logger.error(f"Failed to save history: {str(e)}")

load_history()

# Tone, spam, sentiment, and category keywords
TONE_KEYWORDS = {
    "happy": ["great", "awesome", "happy", "thanks", "excited", "cool", "wonderful", "yay"],
    "urgent": ["urgent", "now", "immediately", "asap", "quick", "hurry", "emergency"],
    "frustrated": ["why", "not working", "problem", "issue", "annoying", "fix", "again"],
    "neutral": ["ok", "fine", "sure", "alright", "yes", "no"]
}
SPAM_KEYWORDS = ["win", "free", "discount", "offer", "click here", "unsubscribe"]
SENTIMENT_KEYWORDS = {
    "positive": ["good", "excellent", "happy", "pleased", "thank"],
    "negative": ["bad", "terrible", "issue", "problem", "sorry"]
}
CATEGORY_KEYWORDS = {
    "promotional": ["sale", "discount", "offer", "deal"],
    "work": ["meeting", "project", "deadline", "task"],
    "personal": ["hi", "hey", "friend", "family"]
}

EMAIL_HEADER = f"\n{company_name}\n----------------------------------------"
EMAIL_FOOTER = f"----------------------------------------\nBest regards,\n{agent_name}\n{company_name}"

# Utility functions
def extract_sender_name(email_from: str) -> str:
    try:
        match = re.search(r"(\w+)\s*(\w+)?", email_from.split('@')[0])
        return match.group(1).capitalize() if match else "Friend"
    except Exception as e:
        logger.error(f"Error extracting sender name from {email_from}: {str(e)}")
        return "Friend"

def clean_header_value(value: str) -> str:
    try:
        return value.replace("\n", "").replace("\r", "") if value else ""
    except Exception as e:
        logger.error(f"Error cleaning header value: {str(e)}")
        return ""

def detect_tone(email_content: str) -> str:
    try:
        content_lower = email_content.lower()
        scores = {tone: 0 for tone in TONE_KEYWORDS}
        for tone, keywords in TONE_KEYWORDS.items():
            scores[tone] = sum(1 for kw in keywords if kw in content_lower)
        return max(scores, key=scores.get, default="neutral")
    except Exception as e:
        logger.error(f"Error detecting tone: {str(e)}")
        return "neutral"

def detect_sentiment(email_content: str) -> str:
    try:
        content_lower = email_content.lower()
        scores = {sentiment: 0 for sentiment in SENTIMENT_KEYWORDS}
        for sentiment, keywords in SENTIMENT_KEYWORDS.items():
            scores[sentiment] = sum(1 for kw in keywords if kw in content_lower)
        return max(scores, key=scores.get, default="neutral")
    except Exception as e:
        logger.error(f"Error detecting sentiment: {str(e)}")
        return "neutral"

def detect_language(email_content: str) -> str:
    try:
        content_lower = email_content.lower()
        if any(word in content_lower for word in ["hola", "gracias", "por favor"]):
            return "Spanish"
        elif any(word in content_lower for word in ["bonjour", "merci"]):
            return "French"
        return "English"
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        return "English"

def detect_spam(email_content: str, sender: str) -> bool:
    try:
        content_lower = email_content.lower()
        spam_score = sum(1 for kw in SPAM_KEYWORDS if kw in content_lower)
        if model:
            prompt = f"Analyze this email content: '{email_content}' from {sender}. Is it spam? Return 'yes' or 'no'."
            response = model.generate_content(prompt)
            return response.text.strip().lower() == "yes"
        return spam_score > 2 or "unsubscribe" in content_lower
    except Exception as e:
        logger.error(f"Error detecting spam: {str(e)}")
        return False

def categorize_email(email_content: str) -> str:
    try:
        content_lower = email_content.lower()
        scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            scores[cat] = sum(1 for kw in keywords if kw in content_lower)
        category = max(scores, key=scores.get, default="general")
        if model:
            prompt = f"Categorize this email: '{email_content}'. Options: promotional, work, personal, general. Return one."
            response = model.generate_content(prompt)
            category = response.text.strip().lower()
        return category
    except Exception as e:
        logger.error(f"Error categorizing email: {str(e)}")
        return "general"

def analyze_behavior(sender_email: str, current_time: datetime) -> Dict:
    try:
        if sender_email not in email_history["contacts"]:
            return {"frequency": "new", "avg_response_time": 0, "tone": "neutral", "context": {}, "priority_score": 0, "sentiment": "neutral"}
        info = email_history["contacts"][sender_email]
        time_diff = (current_time - info["last_contact"]).total_seconds() / 3600
        frequency = "high" if time_diff < 1 else "medium" if time_diff < 24 else "low"
        priority_score = (10 if frequency == "high" else 5 if frequency == "medium" else 1) + len(info["tone_history"]) + (5 if "urgent" in info["tone_history"] else 0)
        email_history["analytics"]["busiest_contacts"][sender_email] = email_history["analytics"]["busiest_contacts"].get(sender_email, 0) + 1
        return {
            "frequency": frequency,
            "avg_response_time": info["avg_response_time"],
            "tone": max(set(info["tone_history"]), key=info["tone_history"].count) if info["tone_history"] else "neutral",
            "context": info["context"],
            "priority_score": priority_score,
            "sentiment": detect_sentiment(" ".join([msg["content"] for msg in info["conversation_history"]]))
        }
    except Exception as e:
        logger.error(f"Error analyzing behavior for {sender_email}: {str(e)}")
        return {"frequency": "new", "avg_response_time": 0, "tone": "neutral", "context": {}, "priority_score": 0, "sentiment": "neutral"}

def update_context(sender_email: str, email_content: str, attachments: List[Dict] = []) -> None:
    try:
        if sender_email not in email_history["contacts"]:
            email_history["contacts"][sender_email] = {
                "name": extract_sender_name(sender_email), "last_contact": datetime.now(),
                "interactions": 0, "tone_history": [], "avg_response_time": 0, "conversation_history": [],
                "context": {}, "tags": []
            }
        info = email_history["contacts"][sender_email]
        info["interactions"] += 1
        info["last_contact"] = datetime.now()
        info["tone_history"].append(detect_tone(email_content))
        info["conversation_history"].append({"role": "user", "content": email_content})
        if info["interactions"] > 1:
            info["avg_response_time"] = ((info["avg_response_time"] * (info["interactions"] - 1)) + (datetime.now() - info["last_contact"]).total_seconds()) / info["interactions"]
        
        context = info.get("context", {})
        content_lower = email_content.lower()
        if "meeting" in content_lower or "schedule" in content_lower:
            context["topic"] = "scheduling"
            date_match = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(st|nd|rd|th)?)", content_lower)
            context["date"] = date_match.group(0) if date_match else None
            info["tags"].append("schedule")
        elif "product" in content_lower or "service" in content_lower:
            context["topic"] = "product_inquiry"
            info["tags"].append("inquiry")
        elif "problem" in content_lower or "issue" in content_lower:
            context["topic"] = "support"
            info["tags"].append("support")
        elif attachments:
            context["topic"] = "attachment_review"
            context["attachments"] = [att["filename"] for att in attachments]
            info["tags"].append("attachment")
        else:
            context["topic"] = "general"
        info["context"] = context
        for tag in info["tags"]:
            email_history["tags"][tag].append(info["conversation_history"][-1]["message_id"] if "message_id" in info["conversation_history"][-1] else hashlib.md5(email_content.encode()).hexdigest())
    except Exception as e:
        logger.error(f"Error updating context for {sender_email}: {str(e)}")

def process_attachments(email_message) -> List[Dict]:
    attachments = []
    try:
        if not email_message.is_multipart():
            return attachments
        for part in email_message.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            filename = part.get_filename()
            if not filename:
                continue
            data = part.get_payload(decode=True)
            if len(data) > 5 * 1024 * 1024:  # 5MB limit
                logger.warning(f"Attachment {filename} exceeds 5MB, skipping.")
                continue
            attachment = {"filename": filename, "data": base64.b64encode(data).decode('utf-8')}
            if pytesseract and Image and BytesIO and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(BytesIO(data))
                    text = pytesseract.image_to_string(img)
                    attachment["ocr_text"] = text
                    logger.info(f"Successfully processed OCR for attachment {filename}.")
                except Exception as e:
                    logger.error(f"Error processing OCR for attachment {filename}: {str(e)}")
                    attachment["ocr_text"] = "OCR failed due to processing error."
            else:
                logger.info(f"Skipping OCR for {filename}: Tesseract or required libraries not available.")
                attachment["ocr_text"] = "OCR not available."
            attachments.append(attachment)
    except Exception as e:
        logger.error(f"Error processing attachments: {str(e)}")
    return attachments

# Gemini functions
def gemini_generate_reply(email_content: str, sender_email: str, sender_name: str, thread_id: Optional[str] = None, attachments: List[Dict] = []) -> List[str]:
    try:
        if not model:
            logger.warning("Gemini API not available, falling back to default reply.")
            return [f"Dear {sender_name}, I’m processing your request with utmost care. Please bear with me."]
        tone = detect_tone(email_content)
        sentiment = detect_sentiment(email_content)
        language = detect_language(email_content)
        behavior = analyze_behavior(sender_email, datetime.now())
        conversation = email_history["contacts"].get(sender_email, {}).get("conversation_history", [])[-5:]
        attachment_context = " ".join([att.get("ocr_text", "") for att in attachments if "ocr_text" in att])
        prompt = (
            f"You are {agent_name}, a superhuman AI from {company_name}. Respond to {sender_name} ({sender_email}) in {language}. "
            f"History:\n" + "\n".join(f"{msg['role']}: {msg['content']}" for msg in conversation) +
            f"\nLatest email: '{email_content}'\nTone: {tone}\nSentiment: {sentiment}\nFrequency: {behavior['frequency']}\n"
            f"Attachments OCR: '{attachment_context}'\n"
            f"Generate 3 concise, professional reply options (each 2-4 sentences) that:\n"
            f"- Avoid repetition from previous replies.\n"
            f"- Match tone, sentiment, and adapt to behavior.\n"
            f"- Use deep contextual understanding and attachment insights.\n"
            f"- Reflect superhuman insight and empathy.\n"
            f"Do not include subject lines in the body."
        )
        response = model.generate_content(prompt)
        replies = response.text.strip().split("\n\n")
        return replies[:3] if len(replies) >= 3 else [replies[0]] * 3 if replies else ["Default reply"]
    except Exception as e:
        logger.error(f"Gemini reply generation failed for {sender_email}: {str(e)}")
        return [f"Dear {sender_name}, I’m processing your request with utmost care. Please bear with me."]

def gemini_summarize_email(email_content: str) -> str:
    try:
        if not model:
            logger.warning("Gemini API not available, falling back to basic summary.")
            return " ".join(email_content.split()[:20]) + "..."
        prompt = f"Summarize this email in 1-2 sentences: '{email_content}'"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini summarize failed: {str(e)}")
        return "Summary unavailable due to processing error."

def gemini_write_email(draft: str) -> str:
    try:
        if not model:
            logger.warning("Gemini API not available, falling back to default draft.")
            return "Dear Recipient, I’m drafting this for you. Please provide more details if needed."
        prompt = f"Write a professional email draft (2-4 sentences) based on: '{draft}'. Do not include a subject line in the body."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini write failed: {str(e)}")
        return "Dear Recipient, I’m drafting this for you. Please provide more details if needed."

def gemini_generate_email_ideas() -> str:
    try:
        if not model:
            logger.warning("Gemini API not available, returning default ideas.")
            return "1. Follow-up on a meeting.\n2. Thank you email for a recent interaction.\n3. Request for feedback on a project."
        prompt = "Generate 3 professional email ideas (1 sentence each) for a business context."
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini generate email ideas failed: {str(e)}")
        return "1. Follow-up on a meeting.\n2. Thank you email for a recent interaction.\n3. Request for feedback on a project."

def predict_send_time(sender_email: str) -> datetime:
    try:
        if sender_email in email_history["contacts"]:
            info = email_history["contacts"][sender_email]
            times = [e["timestamp"] for e in email_history["sent_emails"] if e["to"] == sender_email]
            if times:
                avg_hour = sum(t.hour for t in times) // len(times)
                now = datetime.now()
                scheduled = now.replace(hour=avg_hour, minute=0, second=0)
                return scheduled + timedelta(days=1) if now.hour > avg_hour else scheduled
        return datetime.now() + timedelta(hours=1)
    except Exception as e:
        logger.error(f"Error predicting send time for {sender_email}: {str(e)}")
        return datetime.now() + timedelta(hours=1)

# Email sending function
def send_email(from_email: str, to_email: str, subject: str, body: str, thread_id: Optional[str] = None, message_id: Optional[str] = None, schedule_time: Optional[Union[datetime, str]] = None, forward: bool = False, attachments: List[Dict] = []) -> Dict:
    try:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", to_email):
            logger.error(f"Invalid email address: {to_email}")
            monitor_logs.put(f"Invalid email address: {to_email}")
            return {"status": "error", "message": "Invalid email address."}

        parsed_schedule_time = None
        if isinstance(schedule_time, str):
            try:
                parsed_schedule_time = datetime.fromisoformat(schedule_time)
            except ValueError as e:
                logger.error(f"Invalid schedule_time format: {schedule_time}. Expected ISO format (e.g., '2025-04-08T12:00:00'). Error: {str(e)}")
                monitor_logs.put(f"Invalid schedule_time format: {schedule_time}")
                return {"status": "error", "message": f"Invalid schedule_time format: {str(e)}. Expected ISO format (e.g., '2025-04-08T12:00:00')."}
        elif isinstance(schedule_time, datetime):
            parsed_schedule_time = schedule_time

        sender_name = extract_sender_name(to_email)
        email_body = f"{EMAIL_HEADER}\n\nDear {sender_name},\n\n{body}\n\n{EMAIL_FOOTER}"
        
        msg = EmailMessage()
        msg['Subject'] = clean_header_value(subject) or "No Subject"
        msg['From'] = from_email
        msg['To'] = forward_email if forward else to_email
        msg['Message-ID'] = message_id or f"msg_{int(time.time())}"
        msg['In-Reply-To'] = thread_id or f"thread_{int(time.time())}"
        msg.set_content(email_body)

        recent_emails = [e for e in email_history["sent_emails"] if (datetime.now() - e["timestamp"]).total_seconds() < 60]
        if len(recent_emails) >= 5:
            parsed_schedule_time = predict_send_time(to_email)
            logger.info(f"Rate limit hit, scheduling email for {parsed_schedule_time} to {to_email}.")
            monitor_logs.put(f"Rate limit hit, scheduling email for {parsed_schedule_time} to {to_email}")

        if parsed_schedule_time and parsed_schedule_time > datetime.now():
            email_history["scheduled_emails"].append({
                "to": to_email, "subject": subject, "body": email_body, "time": parsed_schedule_time,
                "thread_id": thread_id, "message_id": msg['Message-ID']
            })
            save_history()
            monitor_logs.put(f"Scheduled email for {to_email} at {parsed_schedule_time}")
            return {"status": "scheduled", "message": f"Scheduled for {parsed_schedule_time}", "thread_id": thread_id, "message_id": msg['Message-ID']}

        for attempt in range(3):
            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
                    smtp.login(from_email, your_app_password)
                    smtp.send_message(msg)
                email_history["sent_emails"].append({
                    "to": to_email, "subject": subject, "body": body, "timestamp": datetime.now(),
                    "thread_id": thread_id, "message_id": msg['Message-ID']
                })
                email_history["analytics"]["total_sent"] += 1
                if thread_id:
                    email_history["threads"][thread_id] = {
                        "participants": [from_email, to_email], "last_message": datetime.now(),
                        "active": True, "message_ids": email_history["threads"].get(thread_id, {}).get("message_ids", []) + [msg['Message-ID']],
                        "summary": gemini_summarize_email(body) if model else "Recent message"
                    }
                save_history()
                logger.info(f"Email sent successfully to {to_email} with subject '{subject}'.")
                monitor_logs.put(f"Email sent successfully to {to_email} with subject '{subject}'")
                return {"status": "success", "message": "Email sent!", "thread_id": thread_id, "message_id": msg['Message-ID']}
            except Exception as e:
                logger.warning(f"SMTP attempt {attempt + 1}/3 failed for {to_email}: {str(e)}")
                monitor_logs.put(f"SMTP attempt {attempt + 1}/3 failed for {to_email}: {str(e)}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    except Exception as e:
        logger.error(f"Error sending email to {to_email}: {str(e)}")
        monitor_logs.put(f"Error sending email to {to_email}: {str(e)}")
        return {"status": "error", "message": str(e)}

# Fetch emails
def fetch_emails(criteria: str = "UNSEEN") -> List[Dict]:
    try:
        for attempt in range(3):
            try:
                mail = imaplib.IMAP4_SSL("imap.gmail.com", timeout=10)
                mail.login(your_email, your_app_password)
                mail.select("inbox")
                status, data = mail.search(None, criteria)
                if status != "OK":
                    raise ValueError("IMAP search failed")
                email_ids = data[0].split()
                emails = []
                for email_id in email_ids[-10:]:
                    status, msg_data = mail.fetch(email_id, "(RFC822)")
                    if status != "OK":
                        logger.warning(f"Failed to fetch email ID {email_id}.")
                        monitor_logs.put(f"Failed to fetch email ID {email_id}")
                        continue
                    raw_email = msg_data[0][1]
                    email_message = email.message_from_bytes(raw_email)
                    body = ""
                    if email_message.is_multipart():
                        for part in email_message.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                                break
                    else:
                        body = email_message.get_payload(decode=True).decode("utf-8", errors="ignore")
                    attachments = process_attachments(email_message)
                    sender = email_message.get("From", "Unknown")
                    sender_email = re.search(r"<(.+?)>", sender) or sender
                    sender_email = sender_email.group(1) if isinstance(sender_email, re.Match) else sender
                    behavior = analyze_behavior(sender_email, datetime.now())
                    email_data = {
                        "from": sender,
                        "subject": email_message.get("Subject", "No Subject"),
                        "body": body.strip(),
                        "thread_id": email_message.get("In-Reply-To") or f"thread_{int(time.time())}",
                        "message_id": email_message.get("Message-ID") or f"msg_{int(time.time())}",
                        "attachments": attachments,
                        "last_contact": datetime.now(),
                        "category": categorize_email(body),
                        "is_spam": detect_spam(body, sender),
                        "priority_score": behavior["priority_score"],
                        "tags": ["urgent"] if "urgent" in body.lower() else []
                    }
                    if email_data["is_spam"]:
                        email_history["spam_emails"].append(email_data)
                        monitor_logs.put(f"Detected spam email from {sender}")
                    else:
                        emails.append(email_data)
                        email_history["categories"][email_data["category"]] = email_history["categories"].get(email_data["category"], []) + [email_data["message_id"]]
                    if criteria == "UNSEEN":
                        mail.store(email_id, "+FLAGS", "\\Seen")
                mail.logout()
                logger.info(f"Fetched {len(emails)} emails with criteria {criteria}.")
                monitor_logs.put(f"Fetched {len(emails)} emails with criteria {criteria}")
                return sorted(emails, key=lambda x: x["priority_score"], reverse=True)
            except Exception as e:
                logger.warning(f"IMAP attempt {attempt + 1}/3 failed: {str(e)}")
                monitor_logs.put(f"IMAP attempt {attempt + 1}/3 failed: {str(e)}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    except Exception as e:
        logger.error(f"Error fetching emails with criteria {criteria}: {str(e)}")
        monitor_logs.put(f"Error fetching emails with criteria {criteria}: {str(e)}")
        return []

# Rules and processing
def apply_rules(email_data: Dict) -> Optional[Dict]:
    try:
        for rule_id, rule in email_history["rules"].items():
            condition, action = rule["condition"], rule["action"]
            if condition(email_data):
                return action(email_data)
        return None
    except Exception as e:
        logger.error(f"Error applying rules to email from {email_data.get('from', 'unknown')}: {str(e)}")
        monitor_logs.put(f"Error applying rules to email from {email_data.get('from', 'unknown')}: {str(e)}")
        return None

def process_emails() -> Dict:
    try:
        unread_emails = fetch_emails("UNSEEN")
        all_emails = fetch_emails("ALL")
        results = {"processed": [], "reminders": [], "analytics": email_history["analytics"], "insights": {}}
        
        for email_data in unread_emails:
            if email_data["is_spam"]:
                logger.info(f"Skipping spam email from {email_data['from']}.")
                monitor_logs.put(f"Skipping spam email from {email_data['from']}")
                continue
            from_email = re.search(r"<(.+?)>", email_data["from"]) or email_data["from"]
            from_email = from_email.group(1) if isinstance(from_email, re.Match) else from_email
            rule_action = apply_rules(email_data)
            if rule_action:
                results["processed"].append(rule_action)
                logger.info(f"Applied rule to email from {from_email}: {rule_action}")
                monitor_logs.put(f"Applied rule to email from {from_email}: {rule_action['message']}")
                continue
            replies = gemini_generate_reply(email_data["body"], from_email, extract_sender_name(from_email), email_data["thread_id"], email_data["attachments"]) if model else [email_history["templates"]["thanks"].format(name=extract_sender_name(from_email), company=company_name)]
            result = send_email(your_email, from_email, email_data["subject"], replies[0], email_data["thread_id"], email_data["message_id"], attachments=email_data["attachments"])
            results["processed"].append({"to": from_email, "subject": email_data["subject"], "reply": result, "options": replies[1:]})
            logger.info(f"Processed email from {from_email}: {result['status']}")
            monitor_logs.put(f"Processed email from {from_email}: {result['message']}")
            notification_queue.put(f"Replied to email from {from_email}: {result['message']}")
        
        unread_count = len(unread_emails)
        spam_count = len(email_history["spam_emails"])
        new_count = sum(1 for e in all_emails if (datetime.now() - e["last_contact"]).total_seconds() < 300)
        if unread_count > 0:
            results["reminders"].append(f"You have {unread_count} unread email{'s' if unread_count > 1 else ''}.")
            monitor_logs.put(f"Found {unread_count} unread emails")
        if new_count > 0:
            results["reminders"].append(f"You have {new_count} new email{'s' if new_count > 1 else ''} in the last 5 minutes.")
            monitor_logs.put(f"Found {new_count} new emails in the last 5 minutes")
        if spam_count > 0:
            results["reminders"].append(f"You have {spam_count} spam email{'s' if spam_count > 1 else ''} detected.")
            monitor_logs.put(f"Detected {spam_count} spam emails")
        results["insights"]["top_contact"] = max(email_history["analytics"]["busiest_contacts"], key=email_history["analytics"]["busiest_contacts"].get, default="None")
        
        save_history()
        logger.info("Email processing completed successfully.")
        monitor_logs.put("Email processing completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error processing emails: {str(e)}")
        monitor_logs.put(f"Error processing emails: {str(e)}")
        return {"processed": [], "reminders": [f"Error processing emails: {str(e)}"], "analytics": email_history["analytics"], "insights": {}}

# Background email monitoring
notification_queue = queue.Queue()
monitor_logs = queue.Queue()
monitor_thread = None
monitor_running = False

def email_monitor_loop(interval: int = 5):
    global monitor_running
    logger.info("Starting email monitor in background...")
    monitor_logs.put("Starting email monitor in background")
    def check_emails():
        while monitor_running:
            try:
                results = process_emails()
                for reminder in results["reminders"]:
                    notification_queue.put(reminder)
                for insight_key, insight_value in results["insights"].items():
                    notification_queue.put(f"Insight: {insight_key} = {insight_value}")
                    monitor_logs.put(f"Insight: {insight_key} = {insight_value}")
                scheduled = email_history["scheduled_emails"][:]
                for email in scheduled:
                    if datetime.now() >= email["time"]:
                        send_result = send_email(your_email, email["to"], email["subject"], email["body"], email["thread_id"], email["message_id"])
                        email_history["scheduled_emails"].remove(email)
                        notification_queue.put(f"Scheduled email sent to {email['to']}: {send_result['message']}")
                        monitor_logs.put(f"Scheduled email sent to {email['to']}: {send_result['message']}")
                save_history()
            except Exception as e:
                logger.error(f"Error in email monitor loop: {str(e)}")
                monitor_logs.put(f"Error in email monitor loop: {str(e)}")
            time.sleep(interval)
    
    global monitor_thread
    monitor_thread = threading.Thread(target=check_emails, daemon=True)
    monitor_thread.start()

# FastAPI setup
app = FastAPI(title="Email Automation Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests
class EmailRequest(BaseModel):
    to_email: str
    subject: str
    body: str
    thread_id: Optional[str] = None
    schedule_time: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

class CredentialsRequest(BaseModel):
    your_email: str
    your_app_password: str
    company_name: str
    agent_name: str
    forward_email: str

@app.get("/")
async def root():
    return {"message": "Email Automation Server is running. Use /docs for API details."}

@app.post("/save-credentials")
async def save_credentials(request: CredentialsRequest):
    global your_email, your_app_password, company_name, agent_name, forward_email, EMAIL_HEADER, EMAIL_FOOTER
    try:
        your_email = request.your_email
        your_app_password = request.your_app_password
        company_name = request.company_name
        agent_name = request.agent_name
        forward_email = request.forward_email
        EMAIL_HEADER = f"\n{company_name}\n----------------------------------------"
        EMAIL_FOOTER = f"----------------------------------------\nBest regards,\n{agent_name}\n{company_name}"
        logger.info("SMTP credentials updated successfully.")
        monitor_logs.put("SMTP credentials updated successfully")
        return {"message": "Credentials saved successfully!"}
    except Exception as e:
        logger.error(f"Failed to save credentials: {str(e)}")
        monitor_logs.put(f"Failed to save credentials: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save credentials: {str(e)}")

@app.post("/start-monitor")
async def start_monitor():
    global monitor_running
    if monitor_running:
        monitor_logs.put("Monitor is already running")
        return {"message": "Monitor is already running."}
    if not your_email or not your_app_password:
        monitor_logs.put("SMTP credentials not set")
        raise HTTPException(status_code=400, detail="SMTP credentials not set. Please save credentials first.")
    if not model:
        monitor_logs.put("Gemini API not available")
        raise HTTPException(status_code=500, detail="Gemini API not available. Cannot start monitor.")
    monitor_running = True
    email_monitor_loop()
    return {"message": "Email monitor started."}

@app.post("/stop-monitor")
async def stop_monitor():
    global monitor_running, monitor_thread
    if not monitor_running:
        monitor_logs.put("Monitor is not running")
        return {"message": "Monitor is not running."}
    monitor_running = False
    if monitor_thread:
        monitor_thread.join()
    return {"message": "Email monitor stopped."}

@app.get("/monitor-logs")
async def get_monitor_logs():
    logs = []
    while not monitor_logs.empty():
        logs.append(monitor_logs.get())
    return {"logs": logs}

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    message = request.message.lower()
    try:
        if not model:
            monitor_logs.put("Gemini API not available for chat")
            return {"response": "Gemini API not available. Please try again later."}
        if "write an email" in message:
            draft = message.replace("write an email", "").strip()
            response = gemini_write_email(draft)
        elif "email ideas" in message:
            response = gemini_generate_email_ideas()
        else:
            prompt = f"You are {agent_name}, a superhuman AI from {company_name}. Respond to: '{message}' in a professional and helpful manner."
            response = model.generate_content(prompt).text.strip()
        logger.info(f"Chat response generated for message: {message}")
        monitor_logs.put(f"Chat response generated for message: {message}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        monitor_logs.put(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

@app.post("/send-email")
async def send_email_endpoint(request: EmailRequest):
    if not your_email or not your_app_password:
        monitor_logs.put("SMTP credentials not set for sending email")
        raise HTTPException(status_code=400, detail="SMTP credentials not set. Please save credentials first.")
    try:
        result = send_email(
            from_email=your_email,
            to_email=request.to_email,
            subject=request.subject,
            body=request.body,
            thread_id=request.thread_id,
            schedule_time=request.schedule_time
        )
        return result
    except Exception as e:
        logger.error(f"Error in send-email endpoint: {str(e)}")
        monitor_logs.put(f"Error in send-email endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fetch-emails")
async def fetch_emails_endpoint(criteria: str = "UNSEEN"):
    if not your_email or not your_app_password:
        monitor_logs.put("SMTP credentials not set for fetching emails")
        raise HTTPException(status_code=400, detail="SMTP credentials not set. Please save credentials first.")
    try:
        emails = fetch_emails(criteria)
        return {"emails": emails}
    except Exception as e:
        logger.error(f"Error in fetch-emails endpoint: {str(e)}")
        monitor_logs.put(f"Error in fetch-emails endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notifications")
async def get_notifications():
    notifications = []
    while not notification_queue.empty():
        notifications.append(notification_queue.get())
    return {"notifications": notifications}

@app.get("/history")
async def get_history():
    try:
        return email_history
    except Exception as e:
        logger.error(f"Error in history endpoint: {str(e)}")
        monitor_logs.put(f"Error in history endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
