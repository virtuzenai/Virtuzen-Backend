import os
import json
import datetime
import time
import threading
import requests
import sqlite3
import logging.handlers
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from cryptography.fernet import Fernet
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from transformers import pipeline
from threading import Lock
import jwt
from functools import wraps

# --- Configuration ---
DB_FILE = "scheduler.db"
LOG_FILE = "app.log"
KEY_FILE = "encryption.key"
GEMINI_API_KEY = "AIzaSyCcAK6ZAmKm1EZmclShAj0_7xZkqZK2kNM"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
JWT_SECRET = os.getenv("JWT_SECRET", "Vt2lCVXOtXU8WVe8aQQftr0TrDdNmnT/wqFv3Ilp0mI=")
DEBUG_MODE = False

# --- Logging Setup ---
logger = logging.getLogger("Scheduler")
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1048576, backupCount=5)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# --- Thread Safety ---
db_lock = Lock()

# --- Initialize Encryption Key ---
def init_encryption_key():
    try:
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, 'rb') as f:
                return f.read()
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as f:
            f.write(key)
        logger.info("Encryption key generated.")
        return key
    except Exception as e:
        logger.error(f"Failed to initialize encryption key: {e}")
        raise SystemExit("Encryption key initialization failed.")

ENCRYPTION_KEY = init_encryption_key()
cipher = Fernet(ENCRYPTION_KEY)

# --- Database Setup ---
def init_db():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    username TEXT,
                    description TEXT,
                    deadline TEXT,
                    priority TEXT,
                    category TEXT,
                    reminders TEXT,
                    completed INTEGER,
                    recurring INTEGER,
                    recurrence_pattern TEXT,
                    cost REAL,
                    estimated_duration REAL,
                    created_at TEXT,
                    tags TEXT,
                    notes TEXT,
                    dependencies TEXT,
                    progress REAL,
                    assigned_to TEXT,
                    time_block TEXT,
                    custom_alert TEXT,
                    energy_level INTEGER,
                    FOREIGN KEY (username) REFERENCES users(username)
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    username TEXT,
                    description TEXT,
                    duration REAL,
                    created_at TEXT,
                    deadline TEXT
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    name TEXT PRIMARY KEY,
                    template TEXT
                )""")
            conn.commit()
        logger.info("Database initialized.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise

init_db()

# --- Pydantic Models for Validation ---
class User(BaseModel):
    username: str
    password: str

class Task(BaseModel):
    description: str
    deadline: str | None = None
    priority: str = "medium"
    category: str = "general"
    reminders: list[int] = [300]
    completed: bool = False
    recurring: bool = False
    recurrence_pattern: str | None = None
    cost: float = 0.0
    estimated_duration: float = 60.0
    tags: list[str] = []
    notes: str = ""
    dependencies: list[str] = []
    progress: float = 0.0
    assigned_to: str
    time_block: str | None = None
    custom_alert: str | None = None
    energy_level: int = 5

# --- ML Models ---
# TensorFlow Model for Duration/Deadline Prediction
def build_tf_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

tf_duration_model = build_tf_model()
tf_deadline_model = build_tf_model()

# PyTorch Model for Adaptive Prioritization
class PriorityNet(nn.Module):
    def __init__(self):
        super(PriorityNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input: duration, cost, urgency, energy
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output: priority score

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

torch_priority_model = PriorityNet()
optimizer = torch.optim.Adam(torch_priority_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# --- NLP Setup ---
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- Authentication Decorator ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        try:
            jwt.decode(token.replace("Bearer ", ""), JWT_SECRET, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

# --- Task Scheduler ---
class TaskSchedulerAI:
    def __init__(self):
        self.tasks = {}
        self.preferences = defaultdict(lambda: {"work_hours": (9, 17), "mood": "normal", "energy_level": 5})
        self.recurring_tasks = defaultdict(list)
        self.task_history = []
        self.user_points = defaultdict(int)
        self.streak = defaultdict(int)
        self.offline_queue = deque()
        self.load_data()
        threading.Thread(target=self.check_reminders, daemon=True).start()
        threading.Thread(target=self.optimize_schedule_periodically, daemon=True).start()
        threading.Thread(target=self.process_offline_queue, daemon=True).start()

    def load_data(self):
        with db_lock:
            try:
                with sqlite3.connect(DB_FILE) as conn:
                    tasks_df = pd.read_sql_query("SELECT * FROM tasks", conn)
                    self.tasks = {row["task_id"]: row.to_dict() for _, row in tasks_df.iterrows()}
                    self.task_history = pd.read_sql_query("SELECT * FROM history", conn).to_dict("records")
                    templates_df = pd.read_sql_query("SELECT * FROM templates", conn)
                    self.templates = {row["name"]: json.loads(row["template"]) for _, row in templates_df.iterrows()}
            except sqlite3.Error as e:
                logger.error(f"Error loading data: {e}")

    def save_task(self, task_id, task):
        with db_lock:
            try:
                with sqlite3.connect(DB_FILE) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO tasks (task_id, username, description, deadline, priority, category, 
                        reminders, completed, recurring, recurrence_pattern, cost, estimated_duration, created_at, 
                        tags, notes, dependencies, progress, assigned_to, time_block, custom_alert, energy_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (task_id, task["assigned_to"], task["description"], task["deadline"], task["priority"],
                         task["category"], json.dumps(task["reminders"]), int(task["completed"]), int(task["recurring"]),
                         task["recurrence_pattern"], task["cost"], task["estimated_duration"], task["created_at"],
                         json.dumps(task["tags"]), task["notes"], json.dumps(task["dependencies"]), task["progress"],
                         task["assigned_to"], task["time_block"], task["custom_alert"], task["energy_level"]))
                    conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Error saving task {task_id}: {e}")
                raise

    def add_task(self, task_data, username):
        try:
            task = Task(**task_data, assigned_to=username).dict()
            task_id = f"{username}_{int(time.time())}"
            task["created_at"] = datetime.datetime.now().isoformat()
            duration = self.predict_duration(task["description"], username)
            deadline = task["deadline"] or self.predict_deadline(task["description"], username)
            task["estimated_duration"] = duration
            task["deadline"] = deadline
            sentiment = sentiment_analyzer(task["description"])[0]
            task["priority"] = "high" if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.8 else task["priority"]
            self.tasks[task_id] = task
            self.assign_time_block(task_id)
            self.save_task(task_id, task)
            logger.info(f"Task added: {task_id}")
            return {"message": f"Task added: {task['description']} (ID: {task_id})", "task_id": task_id}
        except ValidationError as e:
            logger.error(f"Task validation error: {e}")
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Task addition failed: {e}")
            return {"error": "Failed to add task"}, 500

    def predict_duration(self, description, username):
        try:
            history = [h for h in self.task_history if h["username"] == username]
            if not history:
                return 60.0
            df = pd.DataFrame(history)
            X = np.array([len(d.split()) for d in df["description"]]).reshape(-1, 1)
            y = df["duration"].values
            tf_duration_model.fit(X, y, epochs=5, verbose=0)
            return max(15.0, float(tf_duration_model.predict(np.array([[len(description.split())]]))[0]))
        except Exception as e:
            logger.error(f"Duration prediction error: {e}")
            return 60.0

    def predict_deadline(self, description, username):
        try:
            history = [h for h in self.task_history if h["username"] == username and h["deadline"]]
            if not history:
                return (datetime.datetime.now() + datetime.timedelta(days=1)).isoformat()
            df = pd.DataFrame(history)
            df["days_to_deadline"] = df.apply(
                lambda row: (datetime.datetime.fromisoformat(row["deadline"]) - 
                            datetime.datetime.fromisoformat(row["created_at"])).days, axis=1)
            X = np.array([len(d.split()) for d in df["description"]]).reshape(-1, 1)
            y = df["days_to_deadline"].values
            tf_deadline_model.fit(X, y, epochs=5, verbose=0)
            days = max(1, int(tf_deadline_model.predict(np.array([[len(description.split())]]))[0]))
            return (datetime.datetime.now() + datetime.timedelta(days=days)).isoformat()
        except Exception as e:
            logger.error(f"Deadline prediction error: {e}")
            return (datetime.datetime.now() + datetime.timedelta(days=1)).isoformat()

    def complete_task(self, task_id):
        if task_id not in self.tasks:
            return {"error": "Task not found"}, 404
        try:
            task = self.tasks[task_id]
            task["completed"] = True
            task["progress"] = 100
            duration = (datetime.datetime.now() - datetime.datetime.fromisoformat(task["created_at"])).total_seconds() / 60
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute(
                    "INSERT INTO history (task_id, username, description, duration, created_at, deadline) VALUES (?, ?, ?, ?, ?, ?)",
                    (task_id, task["assigned_to"], task["description"], duration, task["created_at"], task["deadline"])
                )
                conn.commit()
            self.user_points[task["assigned_to"]] += 10
            self.streak[task["assigned_to"]] += 1
            self.save_task(task_id, task)
            return {"message": f"Task {task_id} completed. +10 points! Streak: {self.streak[task['assigned_to']]}"}
        except Exception as e:
            logger.error(f"Error completing task {task_id}: {e}")
            return {"error": "Failed to complete task"}, 500

    def list_tasks(self, username, filter_by="all"):
        try:
            tasks = [t for tid, t in self.tasks.items() if tid.startswith(username)]
            if filter_by == "pending":
                tasks = [t for t in tasks if not t["completed"]]
            elif filter_by == "completed":
                tasks = [t for t in tasks if t["completed"]]
            return {"tasks": tasks}
        except Exception as e:
            logger.error(f"List tasks error: {e}")
            return {"error": "Error listing tasks"}, 500

    def optimize_tasks(self):
        try:
            pending = [(tid, t) for tid, t in self.tasks.items() if not t["completed"]]
            current_time = datetime.datetime.now()
            for tid, task in pending:
                if task["deadline"] and current_time > datetime.datetime.fromisoformat(task["deadline"]):
                    task["priority"] = "high"
                features = torch.tensor([task["estimated_duration"], task["cost"], 
                                        1.0 if task["priority"] == "high" else 0.5, 
                                        self.preferences[task["assigned_to"]]["energy_level"]], dtype=torch.float32)
                with torch.no_grad():
                    task["score"] = torch_priority_model(features).item()
            sorted_tasks = sorted(pending, key=lambda x: (-x[1]["score"], x[1]["cost"], x[1]["estimated_duration"]))
            return {"optimized_tasks": [t for _, t in sorted_tasks]}
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {"error": "Optimization failed"}, 500

    def assign_time_block(self, task_id):
        try:
            task = self.tasks[task_id]
            if task["completed"] or task["time_block"]:
                return
            start_hour, end_hour = self.preferences[task["assigned_to"]]["work_hours"]
            duration = task["estimated_duration"] / 60
            available_slots = []
            current_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
            for hour in range(start_hour, end_hour):
                slot_start = current_time.replace(hour=hour)
                if sum(t["estimated_duration"] / 60 for t in self.tasks.values() if t["time_block"] == slot_start.isoformat()) + duration <= 1:
                    available_slots.append(slot_start)
            if available_slots:
                task["time_block"] = random.choice(available_slots).isoformat()
                self.save_task(task_id, task)
        except Exception as e:
            logger.error(f"Time block assignment error: {task_id}: {e}")

    def check_reminders(self):
        while True:
            try:
                current_time = datetime.datetime.now()
                for tid, task in self.tasks.items():
                    if task["completed"] or not task["deadline"]:
                        continue
                    deadline = datetime.datetime.fromisoformat(task["deadline"])
                    for reminder in task["reminders"]:
                        reminder_time = deadline - datetime.timedelta(seconds=reminder)
                        start_hour, end_hour = self.preferences[task["assigned_to"]]["work_hours"]
                        if (current_time >= reminder_time and current_time < deadline and
                            start_hour <= current_time.hour < end_hour):
                            print(f"{task['custom_alert']} (ID: {tid})")
                            break
                time.sleep(60)
            except Exception as e:
                logger.error(f"Reminder check error: {e}")
                time.sleep(60)

    def optimize_schedule_periodically(self):
        while True:
            self.optimize_tasks()
            time.sleep(3600)

    def process_offline_queue(self):
        while True:
            try:
                if self.offline_queue and self.check_internet():
                    action, data = self.offline_queue.popleft()
                    if action == "add_task":
                        self.add_task(data)
                    logger.info("Processed offline queue item.")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Offline queue processing error: {e}")
                time.sleep(10)

    def check_internet(self):
        try:
            requests.get("https://www.google.com", timeout=5)
            return True
        except requests.ConnectionError:
            return False

scheduler = TaskSchedulerAI()
app = Flask(__name__)

# --- API Endpoints ---
@app.route("/register", methods=["POST"])
def register():
    try:
        data = User(**request.json).dict()
        success, message = security.register(data["username"], data["password"])
        if not success:
            return jsonify({"error": message}), 400
        return jsonify({"message": message}), 201
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Register error: {e}")
        return jsonify({"error": "Registration failed"}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = User(**request.json).dict()
        success, message = security.login(data["username"], data["password"])
        if not success:
            return jsonify({"error": message}), 401
        token = jwt.encode({"username": data["username"], "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)}, 
                          JWT_SECRET, algorithm="HS256")
        return jsonify({"message": message, "token": token}), 200
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Login failed"}), 500

@app.route("/tasks", methods=["POST"])
@token_required
def add_task():
    username = jwt.decode(request.headers["Authorization"].replace("Bearer ", ""), JWT_SECRET, algorithms=["HS256"])["username"]
    return jsonify(scheduler.add_task(request.json, username))

@app.route("/tasks", methods=["GET"])
@token_required
def list_tasks():
    username = jwt.decode(request.headers["Authorization"].replace("Bearer ", ""), JWT_SECRET, algorithms=["HS256"])["username"]
    filter_by = request.args.get("filter", "all")
    return jsonify(scheduler.list_tasks(username, filter_by))

@app.route("/tasks/<task_id>", methods=["PUT"])
@token_required
def complete_task(task_id):
    return jsonify(scheduler.complete_task(task_id))

@app.route("/optimize", methods=["GET"])
@token_required
def optimize():
    return jsonify(scheduler.optimize_tasks())

# --- CLI for Pydroid3 ---
def cli_main():
    print("Welcome to Smart Task Scheduler (CLI Mode)!")
    while True:
        username = input("Username: ")
        password = input("Password: ")
        success, message = security.login(username, password)
        print(message)
        if success:
            break
        if "not found" in message and input("Register? (y/n): ").lower() == "y":
            success, reg_message = security.register(username, password)
            print(reg_message)
            if success:
                break
        if input("Try again? (y/n): ").lower() != "y":
            print("Goodbye!")
            return

    print(f"Logged in as {username}. Use 'help' for commands.")
    while True:
        try:
            command = input(f"{username}> ").strip().lower()
            if command == "exit":
                print(f"Goodbye! Points: {scheduler.user_points[username]}, Streak: {scheduler.streak[username]}")
                break
            elif command == "help":
                print("Commands: add <description>, list [all|pending|completed], complete <id>, optimize, exit")
            elif command.startswith("add "):
                task_data = {"description": command[4:]}
                result = scheduler.add_task(task_data, username)
                print(result["message"])
            elif command in ["list", "list all"]:
                result = scheduler.list_tasks(username, "all")
                for task in result["tasks"]:
                    print(f"{task['task_id']}: {task['description']} - {task['progress']}%")
            elif command == "list pending":
                result = scheduler.list_tasks(username, "pending")
                for task in result["tasks"]:
                    print(f"{task['task_id']}: {task['description']} - {task['progress']}%")
            elif command == "list completed":
                result = scheduler.list_tasks(username, "completed")
                for task in result["tasks"]:
                    print(f"{task['task_id']}: {task['description']} - {task['progress']}%")
            elif command.startswith("complete "):
                result = scheduler.complete_task(command[9:])
                print(result["message"])
            elif command == "optimize":
                result = scheduler.optimize()
                for task in result["optimized_tasks"]:
                    print(f"{task['task_id']}: {task['description']} - Time: {task['time_block']}")
            else:
                print("Unknown command. Type 'help' for options.")
        except Exception as e:
            logger.error(f"CLI error: {e}")
            print(f"Error: {e if DEBUG_MODE else 'Something went wrong. Try again.'}")

if __name__ == "__main__":
    if os.getenv("RENDER") == "true":
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
    else:
        cli_main()
