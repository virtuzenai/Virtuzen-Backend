import imaplib
import smtplib
import email
from email.message import EmailMessage
from datetime import datetime, timedelta
import re
import time
import google.generativeai as genai
import logging
import sys
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("email_automation.log")
    ]
)
logger = logging.getLogger(__name__)

# Email credentials from environment variables
your_email = os.getenv("EMAIL_ADDRESS")
your_app_password = os.getenv("EMAIL_APP_PASSWORD")
company_name = os.getenv("COMPANY_NAME", "Workflow Solutions")
agent_name = os.getenv("AGENT_NAME", "Ivan")

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini API initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {str(e)}")
    model = None

# Simulated in-memory database (consider replacing with a database like Redis in production)
email_history = {
    "contacts": {},  # {email: {"name": str, "last_contact": datetime, "interactions": int, "tone_history": list, "avg_response_time": float, "conversation_history": list, "context": dict}}
    "sent_emails": [],  # [{"to": str, "subject": str, "body": str, "timestamp": datetime, "thread_id": str, "message_id": str}]
    "threads": {}  # {thread_id: {"participants": list, "last_message": datetime, "active": bool, "message_ids": list}}
}

# Tone detection keywords (for fallback)
TONE_KEYWORDS = {
    "happy": ["great", "awesome", "happy", "thanks", "excited", "cool", "wonderful", "yay"],
    "urgent": ["urgent", "now", "immediately", "asap", "quick", "hurry", "emergency"],
    "frustrated": ["why", "not working", "problem", "issue", "annoying", "fix", "again"],
    "neutral": ["ok", "fine", "sure", "alright", "yes", "no"]
}

# Professional email format
EMAIL_HEADER = f"""
----------------------------------------
{company_name}
----------------------------------------
"""

EMAIL_FOOTER = f"""
----------------------------------------
Best regards,
{agent_name}
{company_name}
----------------------------------------
"""

# Extract sender name
def extract_sender_name(email_from: str) -> str:
    try:
        match = re.search(r"(\w+)\s*(\w+)?", email_from.split('@')[0])
        return match.group(1).capitalize() if match else "Friend"
    except Exception as e:
        logger.error(f"Error extracting sender name from {email_from}: {str(e)}")
        return "Friend"

# Clean header values
def clean_header_value(value: str) -> str:
    try:
        return value.replace("\n", "").replace("\r", "") if value else ""
    except Exception as e:
        logger.error(f"Error cleaning header value: {str(e)}")
        return ""

# Detect emotional tone (fallback)
def detect_tone(email_content: str) -> str:
    try:
        email_content = email_content.lower()
        scores = {tone: 0 for tone in TONE_KEYWORDS}
        for tone, keywords in TONE_KEYWORDS.items():
            scores[tone] = sum(1 for kw in keywords if kw in email_content)
        return max(scores, key=scores.get, default="neutral")
    except Exception as e:
        logger.error(f"Error detecting tone: {str(e)}")
        return "neutral"

# Analyze recipient behavior and context
def analyze_behavior(sender_email: str, current_time: datetime) -> Dict:
    try:
        if sender_email not in email_history["contacts"]:
            return {"frequency": "new", "avg_response_time": 0, "tone": "neutral", "context": {}}
        
        info = email_history["contacts"][sender_email]
        interactions = info["interactions"]
        last_contact = info["last_contact"]
        tone_history = info["tone_history"]

        time_diff = (current_time - last_contact).total_seconds() / 3600  # Hours
        frequency = "high" if time_diff < 1 else "medium" if time_diff < 24 else "low"
        avg_response_time = info["avg_response_time"] or time_diff
        tone = max(set(tone_history), key=tone_history.count) if tone_history else "neutral"

        return {
            "frequency": frequency,
            "avg_response_time": avg_response_time,
            "tone": tone,
            "context": info["context"]
        }
    except Exception as e:
        logger.error(f"Error analyzing behavior for {sender_email}: {str(e)}")
        return {"frequency": "new", "avg_response_time": 0, "tone": "neutral", "context": {}}

# Update context based on email content
def update_context(sender_email: str, email_content: str) -> None:
    try:
        info = email_history["contacts"][sender_email]
        context = info.get("context", {})

        if any(keyword in email_content.lower() for keyword in ["meeting", "schedule", "appointment"]):
            context["topic"] = "scheduling"
            date_match = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(st|nd|rd|th)?)", email_content.lower())
            context["date"] = date_match.group(0) if date_match else None
        elif any(keyword in email_content.lower() for keyword in ["product", "service", "pricing", "features"]):
            context["topic"] = "product_inquiry"
        elif any(keyword in email_content.lower() for keyword in ["problem", "issue", "not working", "fix"]):
            context["topic"] = "support"
        else:
            context["topic"] = "general"

        info["context"] = context
    except Exception as e:
        logger.error(f"Error updating context for {sender_email}: {str(e)}")

# Generate reply using Gemini API with retry mechanism
def generate_reply(email_content: str, sender_email: str, sender_name: str = "Friend", thread_id: Optional[str] = None) -> str:
    try:
        email_content = re.sub(r"<[^>]+>", "", email_content)  # Remove HTML tags
        email_content = re.sub(r"[{}].*?[!important].*?;", "", email_content)  # Remove CSS
        email_content = email_content.strip()

        now = datetime.now()
        tone = detect_tone(email_content)
        behavior = analyze_behavior(sender_email, now)

        if sender_email not in email_history["contacts"]:
            email_history["contacts"][sender_email] = {
                "name": sender_name,
                "last_contact": now,
                "interactions": 0,
                "tone_history": [],
                "avg_response_time": 0,
                "conversation_history": [],
                "context": {}
            }
        info = email_history["contacts"][sender_email]
        info["interactions"] += 1
        info["last_contact"] = now
        info["tone_history"].append(tone)
        if info["interactions"] > 1:
            info["avg_response_time"] = ((info["avg_response_time"] * (info["interactions"] - 1)) + (now - info["last_contact"]).total_seconds()) / info["interactions"]

        update_context(sender_email, email_content)

        info["conversation_history"].append({"role": "user", "content": email_content})
        conversation = info["conversation_history"][-5:]

        prompt = (
            f"You are a highly professional and empathetic customer service representative named {agent_name}, working for {company_name}. "
            f"Your goal is to respond to emails in a natural, human-like manner, as if you were a top-tier call center agent. "
            f"The recipient is {sender_name} ({sender_email}). "
            f"Here’s the conversation history (last 5 messages):\n"
        )
        for msg in conversation:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += (
            f"\nThe latest email from {sender_name} is: '{email_content}'.\n"
            f"Their detected tone is {tone}, and their reply frequency is {behavior['frequency']}.\n"
            f"Additional context: {behavior['context']}.\n"
            f"Generate a concise, professional reply (2-4 sentences) that:\n"
            f"- Addresses the content of their email with precision and empathy.\n"
            f"- Matches their tone ({tone}) and adapts to their behavior (e.g., more formal for low frequency, warmer for high frequency).\n"
            f"- Avoids repetition from previous replies by using the conversation history.\n"
            f"- Uses a professional tone with a human touch, as if written by a top customer service agent.\n"
            f"- Does not use emojis.\n"
            f"- If asked 'Who are you?', respond with: 'I’m {agent_name} from {company_name}, here to assist you.'\n"
            f"- If the email is about scheduling, confirm or suggest a date and time, or ask for preferences.\n"
            f"- If the email is a product inquiry, provide a helpful response or ask for more details.\n"
            f"- If the email is a support issue, offer a solution or next steps.\n"
            f"- For general queries, provide a thoughtful response or ask clarifying questions.\n"
            f"Return only the reply text."
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if model:
                    response = model.generate_content(prompt)
                    reply = response.text.strip()
                    break
                else:
                    raise ValueError("Gemini API model not initialized.")
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("Max retries reached for Gemini API. Falling back to keyword-based reply.")
                    reply = None
                time.sleep(1)
        else:
            reply = None

        if not reply:
            if "who are you" in email_content.lower():
                reply = f"I’m {agent_name} from {company_name}, here to assist you."
            elif any(keyword in email_content.lower() for keyword in ["meeting", "schedule", "reschedule"]):
                date_match = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(st|nd|rd|th)?)", email_content.lower())
                date = date_match.group(0) if date_match else "a suitable date"
                reply = f"Dear {sender_name}, I’d be happy to arrange that meeting for {date}. Does that time work for you, or would you prefer another date?"
            elif any(keyword in email_content.lower() for keyword in ["question", "help", "assist"]):
                reply = f"Dear {sender_name}, thank you for reaching out. Could you please provide more details so I can assist you better?"
            else:
                reply = f"Dear {sender_name}, thank you for your message. I’ll look into this and get back to you with more details shortly."

        info["conversation_history"].append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"Error generating reply for {sender_email}: {str(e)}")
        return f"Dear {sender_name}, I apologize for the inconvenience. I’m experiencing a technical issue, but I’ll follow up with you shortly."

# Send email function with retry mechanism
def send_email(from_email: str, to_email: str, subject: str, incoming_email: str, thread_id: Optional[str] = None, message_id: Optional[str] = None) -> Dict:
    try:
        sender_name = extract_sender_name(to_email)
        reply = generate_reply(incoming_email, to_email, sender_name, thread_id)

        incoming_email = re.sub(r"<[^>]+>", "", incoming_email)  # Remove HTML tags
        incoming_email = re.sub(r"[{}].*?[!important].*?;", "", incoming_email)  # Remove CSS

        subject = clean_header_value(f"RE: {subject}")
        from_email = clean_header_value(from_email)
        to_email = clean_header_value(to_email)  # Reply to the sender (removed test_recipient)

        if not thread_id:
            thread_id = f"thread_{int(time.time())}"
        if not message_id:
            message_id = f"msg_{int(time.time())}"

        email_history["threads"][thread_id] = {
            "participants": [from_email, to_email],
            "last_message": datetime.now(),
            "active": True,
            "message_ids": email_history["threads"].get(thread_id, {}).get("message_ids", []) + [message_id]
        }

        email_body = f"{EMAIL_HEADER}\n\nDear {sender_name},\n\n{reply}\n\n--- Original Message ---\n{incoming_email}\n\n{EMAIL_FOOTER}"

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Message-ID'] = message_id
        msg['In-Reply-To'] = thread_id
        msg.set_content(email_body)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
                    smtp.login(from_email, your_app_password)
                    smtp.send_message(msg)
                    email_history["sent_emails"].append({
                        "to": to_email,
                        "subject": subject,
                        "body": reply,
                        "timestamp": datetime.now(),
                        "thread_id": thread_id,
                        "message_id": message_id
                    })
                    logger.info(f"Email sent successfully to {to_email} in thread {thread_id}.")
                    return {"status": "success", "message": "Email sent successfully!", "thread_id": thread_id, "message_id": message_id}
            except Exception as e:
                logger.warning(f"SMTP attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("Max retries reached for SMTP. Email sending failed.")
                    return {"status": "error", "message": f"Failed to send email after {max_retries} attempts: {str(e)}"}
                time.sleep(2)
    except Exception as e:
        logger.error(f"Error in send_email for {to_email}: {str(e)}")
        return {"status": "error", "message": f"Failed to send email: {str(e)}"}

# Fetch unread emails with retry mechanism
def fetch_unread_emails() -> List[Dict]:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com", timeout=10)
            mail.login(your_email, your_app_password)
            mail.select("inbox")

            status, data = mail.search(None, "UNSEEN")
            if status != "OK":
                raise ValueError("IMAP search failed.")

            email_ids = data[0].split()
            emails = []
            for email_id in email_ids[:10]:  # Limit to 10 emails per cycle for scalability
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                if status != "OK":
                    logger.warning(f"Failed to fetch email ID {email_id}.")
                    continue

                raw_email = msg_data[0][1]
                email_message = email.message_from_bytes(raw_email)

                from_ = email_message.get("From", "Unknown")
                subject = email_message.get("Subject", "No Subject")
                thread_id = email_message.get("In-Reply-To") or f"thread_{int(time.time())}"
                message_id = email_message.get("Message-ID") or f"msg_{int(time.time())}"
                body = ""
                if email_message.is_multipart():
                    for part in email_message.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            break
                else:
                    body = email_message.get_payload(decode=True).decode("utf-8", errors="ignore")

                emails.append({"from": from_, "subject": subject, "body": body.strip(), "thread_id": thread_id, "message_id": message_id})
                mail.store(email_id, "+FLAGS", "\\Seen")

            mail.logout()
            logger.info(f"Fetched {len(emails)} unread emails.")
            return emails
        except Exception as e:
            logger.warning(f"IMAP attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached for IMAP. Email fetching failed.")
                return [{"status": "error", "message": f"Failed to fetch emails after {max_retries} attempts: {str(e)}"}]
            time.sleep(2)

# Process emails and reply
def process_emails() -> List[Dict]:
    try:
        emails = fetch_unread_emails()
        results = []
        for email_data in emails:
            if "status" in email_data and email_data["status"] == "error":
                results.append(email_data)
                continue

            from_email = re.search(r"<(.+?)>", email_data["from"])
            from_email = from_email.group(1) if from_email else email_data["from"]
            subject = email_data["subject"]
            body = email_data["body"]
            thread_id = email_data["thread_id"]
            message_id = email_data["message_id"]

            result = send_email(your_email, from_email, subject, body, thread_id, message_id)
            results.append({"to": from_email, "subject": subject, "reply": result, "thread_id": thread_id, "message_id": message_id})
        return results
    except Exception as e:
        logger.error(f"Error processing emails: {str(e)}")
        return [{"status": "error", "message": f"Failed to process emails: {str(e)}"}]

# Continuous monitoring loop
def email_monitor_loop(interval: int = 15) -> None:
    logger.info("Starting email monitor in production mode...")
    while True:
        try:
            start_time = time.time()
            results = process_emails()
            for result in results:
                if "status" in result:
                    logger.error(result["message"])
                else:
                    logger.info(f"Replied to {result['to']} in thread {result['thread_id']}: {result['reply']['message']}")
            elapsed_time = time.time() - start_time
            logger.debug(f"Email processing cycle took {elapsed_time:.2f} seconds.")
            time.sleep(max(0, interval - elapsed_time))
        except Exception as e:
            logger.error(f"Error in email monitor loop: {str(e)}")
            time.sleep(interval)

if __name__ == "__main__":
    email_monitor_loop()
