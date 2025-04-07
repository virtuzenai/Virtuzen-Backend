import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging

from email_automation import process_emails  # Updated import

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global monitor control
monitor_thread = None
monitor_running = False

# Function to repeatedly check emails
def email_monitor_loop(interval=15):
    global monitor_running
    while monitor_running:
        process_emails()  # Updated to call process_emails
        time.sleep(interval)

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.end_headers()

    def send_json_response(self, data):
        response = json.dumps(data).encode("utf-8")
        self._set_headers()
        self.wfile.write(response)

    def do_GET(self):
        if self.path == "/":
            self.send_json_response({"status": "ok", "message": "Email automation server is running"})
        else:
            self.send_json_response({"status": "error", "message": "Invalid endpoint"})

    def do_POST(self):
        global monitor_thread
        global monitor_running

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        logger.info(f"Received POST data: {post_data}")

        if self.path == "/start_monitor":
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
            if monitor_running:
                monitor_running = False
                self.send_json_response({"status": "success", "message": "Email monitor stopped"})
                logger.info("Email monitor stopped.")
            else:
                self.send_json_response({"status": "error", "message": "Monitor is not running"})
                logger.warning("Attempted to stop monitor while not running.")

        else:
            self.send_json_response({"status": "error", "message": "Invalid endpoint"})

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ("", int(os.getenv("PORT", port)))
    httpd = server_class(server_address, handler_class)
    logger.info(f"Starting server on port {server_address[1]}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
