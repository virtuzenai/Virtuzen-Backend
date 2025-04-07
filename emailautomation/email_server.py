import http.server
import socketserver
import json
from urllib.parse import parse_qs
import threading
import logging
from email_automation import send_email, fetch_unread_emails, process_emails, email_history, email_monitor_loop

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("email_server.log")
    ]
)
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", 8000))
monitor_thread = None
monitor_running = False

class EmailHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            params = parse_qs(post_data)

            if self.path == "/send_email":
                to_email = params.get('to_email', [''])[0]
                subject = params.get('subject', [''])[0]
                incoming_email = params.get('incoming_email', [''])[0]
                if not all([to_email, subject, incoming_email]):
                    self.send_error_response(400, "Missing required fields")
                    return
                result = send_email(your_email, to_email, subject, incoming_email)
                self.send_json_response(result, 200 if result["status"] == "success" else 500)

            elif self.path == "/process_emails":
                results = process_emails()
                self.send_json_response(results)

            elif self.path == "/start_monitor":
                global monitor_thread, monitor_running
                if not monitor_running:
                    monitor_thread = threading.Thread(target=email_monitor_loop, args=(15,))
                    monitor_thread.daemon = True
                    monitor_thread.start()
                    monitor_running = True
                    self.send_json_response({"status": "success", "message": "Email monitor started"})
                    logger.info("Email monitor started.")
                else:
                    self.send_json_response({"status": "error", "message": "Monitor already running"})
                    logger.warning("Attempted to start monitor while already running.")

            elif self.path == "/stop_monitor":
                global monitor_running
                if monitor_running:
                    monitor_running = False
                    self.send_json_response({"status": "success", "message": "Email monitor stopped"})
                    logger.info("Email monitor stopped.")
                else:
                    self.send_json_response({"status": "error", "message": "Monitor not running"})
                    logger.warning("Attempted to stop monitor while not running.")
            else:
                self.send_error_response(404, "Not Found")
        except Exception as e:
            logger.error(f"Error handling POST request: {str(e)}")
            self.send_error_response(500, f"Server error: {str(e)}")

    def do_GET(self):
        try:
            if self.path == "/fetch_emails":
                emails = fetch_unread_emails()
                self.send_json_response(emails)
            elif self.path == "/email_history":
                self.send_json_response(email_history)
            else:
                self.send_error_response(404, "Not Found")
        except Exception as e:
            logger.error(f"Error handling GET request: {str(e)}")
            self.send_error_response(500, f"Server error: {str(e)}")

    def send_json_response(self, data, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except Exception as e:
            logger.error(f"Error sending JSON response: {str(e)}")

    def send_error_response(self, code, message):
        try:
            self.send_response(code)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": message}).encode())
        except Exception as e:
            logger.error(f"Error sending error response: {str(e)}")

Handler = EmailHandler
try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        logger.info(f"Serving at port {PORT}")
        httpd.serve_forever()
except Exception as e:
    logger.error(f"Failed to start server: {str(e)}")
    raise
