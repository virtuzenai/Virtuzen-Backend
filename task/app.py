import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
import json
import datetime
import time
import threading
import requests
import sqlite3
import logging
import random
from threading import Lock
from flask import Flask, request, jsonify
from cryptography.fernet import Fernet
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from functools import wraps

# --- Configuration ---
TASK_DB = "tasks.db"
USER_DB = "users.db"
LOG_FILE = "app.log"
KEY_FILE = "encryption.key"
BACKUP_DIR = "backups"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
os.makedirs(BACKUP_DIR, exist_ok=True)

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCcAK6ZAmKm1EZmclShAj0_7xZkqZK2kNM")
SECRET_KEY = os.getenv("SECRET_KEY", "Vt2lCVXOtXU8WVe8aQQftr0TrDdNmnT/wqFv3Ilp0mI=")

# --- Logging Setup ---
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())  # Also log to console for Render

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
        logging.info("Encryption key generated.")
        return key
    except Exception as e:
        logging.error(f"Failed to initialize encryption key: {e}")
        raise SystemExit("Encryption key initialization failed.")

ENCRYPTION_KEY = init_encryption_key()
cipher = Fernet(ENCRYPTION_KEY)

# --- Database Setup ---
def init_db():
    try:
        with sqlite3.connect(TASK_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    deadline TEXT,
                    priority TEXT,
                    category TEXT,
                    reminders TEXT,
                    completed INTEGER,
                    recurring INTEGER,
                    recurrence_pattern TEXT,
                    cost REAL,
                    duration REAL,
                    created_at TEXT,
                    tags TEXT,
                    notes TEXT,
                    dependencies TEXT,
                    progress REAL,
                    assigned_to TEXT,
                    time_block TEXT,
                    custom_alert TEXT,
                    energy_level INTEGER,
                    cluster INTEGER
                )
            """)
        with sqlite3.connect(USER_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT,
                    points INTEGER DEFAULT 0,
                    streak INTEGER DEFAULT 0,
                    energy_log TEXT
                )
            """)
        logging.info("Databases initialized.")
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {e}")
        raise SystemExit("Database initialization failed.")

init_db()

# --- ML Models ---
def build_tf_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

class PriorityNet(nn.Module):
    def __init__(self, input_dim):
        super(PriorityNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- Security Manager ---
class SecurityManager:
    def __init__(self):
        self.users = self.load_users()

    def load_users(self):
        try:
            with sqlite3.connect(USER_DB) as conn:
                users = conn.execute("SELECT username, password FROM users").fetchall()
                return {u[0]: u[1] for u in users}
        except sqlite3.Error as e:
            logging.error(f"Error loading users: {e}")
            return {}

    def save_users(self):
        with db_lock:
            try:
                with sqlite3.connect(USER_DB) as conn:
                    for username, password in self.users.items():
                        conn.execute("INSERT OR REPLACE INTO users (username, password) VALUES (?, ?)", 
                                    (username, password))
                    conn.commit()
                logging.info("Users saved successfully.")
            except sqlite3.Error as e:
                logging.error(f"Error saving portavoz: {e}")
                raise

    def register(self, username, password):
        if not username or not password:
            return False, "Username and password cannot be empty."
        if username in self.users:
            logging.warning(f"Registration attempt for existing user: {username}")
            return False, f"Username '{username}' is already taken."
        try:
            encrypted_pass = cipher.encrypt(password.encode()).decode()
            self.users[username] = encrypted_pass
            self.save_users()
            logging.info(f"User registered: {username}")
            return True, f"User '{username}' registered successfully."
        except Exception as e:
            logging.error(f"Registration error for {username}: {e}")
            return False, f"Registration failed: {e}"

    def login(self, username, password):
        if not username or not password:
            return False, "Username and password cannot be empty."
        if username not in self.users:
            logging.warning(f"Login attempt for unknown user: {username}")
            return False, f"Username '{username}' not found. Please register."
        try:
            decrypted_pass = cipher.decrypt(self.users[username].encode()).decode()
            if decrypted_pass == password:
                logging.info(f"Login success: {username}")
                return True, f"Welcome back, {username}!"
            logging.warning(f"Login failed for {username}: wrong password")
            return False, "Incorrect password."
        except Exception as e:
            logging.error(f"Login error for {username}: {e}")
            return False, "Login error. Please try again."

security = SecurityManager()

# --- Task Scheduler ---
class TaskSchedulerAI:
    def __init__(self):
        self.preferences = {"work_hours": (9, 17), "default_reminder": 300, "mood": "normal", "energy_level": 5}
        self.recurring_tasks = defaultdict(list)
        self.offline_queue = deque()
        try:
            self.tf_duration_model = build_tf_model(2)
            self.tf_deadline_model = build_tf_model(2)
            self.pt_priority_model = PriorityNet(4)
            self.pt_optimizer = torch.optim.Adam(self.pt_priority_model.parameters(), lr=0.001)
            self.pt_criterion = nn.MSELoss()
            self.load_models()
            threading.Thread(target=self.check_reminders, daemon=True).start()
            threading.Thread(target=self.optimize_schedule_periodically, daemon=True).start()
            threading.Thread(target=self.process_offline_queue, daemon=True).start()
            logging.info("TaskSchedulerAI initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize TaskSchedulerAI: {e}")
            raise

    def load_models(self):
        try:
            if os.path.exists("duration_model.h5"):
                self.tf_duration_model = tf.keras.models.load_model("duration_model.h5")
            if os.path.exists("deadline_model.h5"):
                self.tf_deadline_model = tf.keras.models.load_model("deadline_model.h5")
            if os.path.exists("priority_model.pt"):
                self.pt_priority_model.load_state_dict(torch.load("priority_model.pt"))
            logging.info("Models loaded.")
        except Exception as e:
            logging.error(f"Model loading error: {e}")

    def save_models(self):
        try:
            self.tf_duration_model.save("duration_model.h5")
            self.tf_deadline_model.save("deadline_model.h5")
            torch.save(self.pt_priority_model.state_dict(), "priority_model.pt")
            logging.info("Models saved.")
        except Exception as e:
            logging.error(f"Model saving error: {e}")

    def train_models(self, username):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                df = pd.read_sql_query("SELECT * FROM tasks WHERE assigned_to = ? AND completed = 1", conn, params=(username,))
                if len(df) < 5:
                    return
                X_duration = np.array([[len(row["description"].split()), row["energy_level"]] for _, row in df.iterrows()])
                y_duration = df["duration"].values
                self.tf_duration_model.fit(X_duration, y_duration, epochs=10, batch_size=32, verbose=0)
                df["days_to_deadline"] = df.apply(
                    lambda row: (datetime.datetime.fromisoformat(row["deadline"]) - 
                                datetime.datetime.fromisoformat(row["created_at"])).days, axis=1)
                X_deadline = np.array([[len(row["description"].split()), row["energy_level"]] for _, row in df.iterrows()])
                y_deadline = df["days_to_deadline"].values
                self.tf_deadline_model.fit(X_deadline, y_deadline, epochs=10, batch_size=32, verbose=0)
                X_priority = torch.tensor([[1 if row["priority"] == "high" else 0.5 if row["priority"] == "medium" else 0.1,
                                           row["cost"], row["duration"], row["energy_level"]] 
                                           for _, row in df.iterrows()], dtype=torch.float32)
                y_priority = torch.tensor([row["progress"] / 100 for _, row in df.iterrows()], dtype=torch.float32)
                for _ in range(10):
                    self.pt_optimizer.zero_grad()
                    outputs = self.pt_priority_model(X_priority).squeeze()
                    loss = self.pt_criterion(outputs, y_priority)
                    loss.backward()
                    self.pt_optimizer.step()
                self.save_models()
                logging.info(f"Models trained for {username}.")
            except Exception as e:
                logging.error(f"Model training error: {e}")
            finally:
                conn.close()

    def add_task(self, task_input, username, template=None):
        with db_lock:
            try:
                task_details = self.process_with_gemini(task_input) if not template else self.get_template(template)
                task_details["description"] = task_input if not template else task_details["description"].format(task_input)
                task_id = f"{username}_{int(time.time())}"
                duration = self.predict_duration(task_details["description"], task_details.get("energy_level", 5))
                deadline = task_details.get("deadline") or self.predict_deadline(task_details["description"], username)
                task = {
                    "id": task_id,
                    "description": task_details["description"],
                    "deadline": deadline,
                    "priority": self.adjust_priority(task_details.get("priority", "medium"), task_details["description"]),
                    "category": task_details.get("category", "general"),
                    "reminders": json.dumps(task_details.get("reminders", [self.preferences["default_reminder"]])),
                    "completed": 0,
                    "recurring": int(task_details.get("recurring", False)),
                    "recurrence_pattern": task_details.get("recurrence_pattern", None),
                    "cost": task_details.get("cost", 0),
                    "duration": duration,
                    "created_at": datetime.datetime.now().isoformat(),
                    "tags": json.dumps(self.auto_tag(task_details["description"], task_details.get("tags", []))),
                    "notes": task_details.get("notes", ""),
                    "dependencies": json.dumps(task_details.get("dependencies", [])),
                    "progress": 0,
                    "assigned_to": task_details.get("assigned_to", username),
                    "time_block": None,
                    "custom_alert": task_details.get("custom_alert", f"{task_details['description']} is due soon!"),
                    "energy_level": task_details.get("energy_level", self.preferences["energy_level"]),
                    "cluster": None
                }
                self.assign_time_block(task)
                self.cluster_tasks(username)
                conn = sqlite3.connect(TASK_DB)
                conn.execute("""
                    INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(task.values()))
                conn.commit()
                if task["recurring"]:
                    self.recurring_tasks[task_id].append(task)
                self.train_models(username)
                return f"Task added: {task['description']} (ID: {task_id})"
            except Exception as e:
                logging.error(f"Task addition failed: {e}")
                return f"Error adding task: {e}"
            finally:
                conn.close()

    def process_with_gemini(self, input_text):
        headers = {"Content-Type": "application/json"}
        prompt = (
            f"Parse this task: '{input_text}'. Extract: description, deadline (ISO format), "
            "priority (low, medium, high), category, reminders (list of seconds), recurring (true/false), "
            "recurrence_pattern (daily, weekly, monthly), cost (numeric), tags (list of #hashtags), "
            "notes (after 'note:'), dependencies (list), assigned_to (username), custom_alert (string), "
            "energy_level (1-10), sentiment (positive/negative/neutral). Return as JSON."
        )
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(content.strip("```json\n").strip("```"))
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            self.offline_queue.append(("add_task", input_text))
            return {"description": input_text, "priority": "medium", "category": "general"}

    def adjust_priority(self, base_priority, description):
        try:
            sentiment = self.process_with_gemini(description).get("sentiment", "neutral")
            if sentiment == "negative":
                return "high" if base_priority == "medium" else base_priority
            return base_priority
        except Exception:
            return base_priority

    def auto_tag(self, description, existing_tags):
        try:
            prompt = f"Suggest tags for this task: '{description}'. Return as JSON list of #hashtags."
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            tags = json.loads(content.strip("```json\n").strip("```"))
            return list(set(existing_tags + tags[:3]))
        except Exception:
            return existing_tags

    def predict_duration(self, description, energy_level):
        try:
            X = np.array([[len(description.split()), energy_level]])
            return max(15, int(self.tf_duration_model.predict(X, verbose=0)[0]))
        except Exception as e:
            logging.error(f"Duration prediction error: {e}")
            return 60

    def predict_deadline(self, description, username):
        try:
            X = np.array([[len(description.split()), self.preferences["energy_level"]]])
            days = max(1, int(self.tf_deadline_model.predict(X, verbose=0)[0]))
            return (datetime.datetime.now() + datetime.timedelta(days=days)).isoformat()
        except Exception as e:
            logging.error(f"Deadline prediction error: {e}")
            return (datetime.datetime.now() + datetime.timedelta(days=1)).isoformat()

    def predict_priority(self, task):
        try:
            X = torch.tensor([[1 if task["priority"] == "high" else 0.5 if task["priority"] == "medium" else 0.1,
                              task["cost"], task["duration"], task["energy_level"]]], dtype=torch.float32)
            with torch.no_grad():
                score = self.pt_priority_model(X).item()
            return min(max(score, 0), 1)
        except Exception as e:
            logging.error(f"Priority prediction error: {e}")
            return 0.5

    def complete_task(self, task_id):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
                if not task:
                    return "Task not found."
                conn.execute("UPDATE tasks SET completed = 1, progress = 100 WHERE id = ?", (task_id,))
                conn.commit()
                username = task_id.split('_')[0]
                conn = sqlite3.connect(USER_DB)
                conn.execute("UPDATE users SET points = points + 10, streak = streak + 1 WHERE username = ?", (username,))
                conn.commit()
                task_dict = dict(zip([d[0] for d in conn.execute("PRAGMA table_info(tasks)").fetchall()], task))
                if task_dict["recurring"]:
                    self.reschedule_recurring_task(task_dict)
                self.train_models(username)
                return f"Task {task_id} completed. +10 points!"
            except sqlite3.Error as e:
                logging.error(f"Error completing task {task_id}: {e}")
                return "Failed to complete task."
            finally:
                conn.close()

    def reschedule_recurring_task(self, task):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                deadline = datetime.datetime.fromisoformat(task["deadline"])
                pattern = task["recurrence_pattern"]
                new_deadline = (deadline + datetime.timedelta(days=1 if pattern == "daily" else 7 if pattern == "weekly" else 30)).isoformat()
                new_task = task.copy()
                new_task["id"] = f"{task['assigned_to']}_{int(time.time())}"
                new_task["deadline"] = new_deadline
                new_task["completed"] = 0
                new_task["progress"] = 0
                new_task["created_at"] = datetime.datetime.now().isoformat()
                conn.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                            tuple(new_task.values()))
                conn.commit()
                self.recurring_tasks[task["id"]].append(new_task)
                self.assign_time_block(new_task)
            except sqlite3.Error as e:
                logging.error(f"Error rescheduling task {task['id']}: {e}")
            finally:
                conn.close()
    def check_reminders(self):
        while True:
            with db_lock:
                try:
                    conn = sqlite3.connect(TASK_DB)
                    tasks = conn.execute("SELECT * FROM tasks WHERE completed = 0").fetchall()
                    current_time = datetime.datetime.now()
                    for task in tasks:
                        task_dict = dict(zip([d[0] for d in conn.execute("PRAGMA table_info(tasks)").fetchall()], task))
                        if not task_dict["deadline"]:
                            continue
                        deadline = datetime.datetime.fromisoformat(task_dict["deadline"])
                        reminders = json.loads(task_dict["reminders"])
                        for reminder in reminders:
                            reminder_time = deadline - datetime.timedelta(seconds=reminder)
                            start_hour, end_hour = self.preferences["work_hours"]
                            if (current_time >= reminder_time and current_time < deadline and
                                start_hour <= current_time.hour < end_hour):
                                logging.info(f"Reminder: {task_dict['custom_alert']} (ID: {task_dict['id']})")
                                break
                    time.sleep(60)
                except sqlite3.Error as e:
                    logging.error(f"Reminder check error: {e}")
                    time.sleep(60)
                finally:
                    conn.close()

    def optimize_schedule_periodically(self):
        while True:
            self.optimize_tasks()
            time.sleep(3600)

    def optimize_tasks(self):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                tasks = conn.execute("SELECT * FROM tasks WHERE completed = 0").fetchall()
                current_time = datetime.datetime.now()
                mood_factor = 1 if self.preferences["mood"] == "normal" else 0.5 if self.preferences["mood"] == "low" else 1.5
                energy_factor = self.preferences["energy_level"] / 5
                task_list = [dict(zip([d[0] for d in conn.execute("PRAGMA table_info(tasks)").fetchall()], t)) for t in tasks]
                for task in task_list:
                    if task["deadline"] and current_time > datetime.datetime.fromisoformat(task["deadline"]):
                        task["priority"] = "high"
                        conn.execute("UPDATE tasks SET priority = 'high' WHERE id = ?", (task["id"],))
                    task["score"] = self.predict_priority(task) * mood_factor * energy_factor
                sorted_tasks = sorted(
                    task_list,
                    key=lambda x: (
                        not self.check_dependencies(x["id"]),
                        -x["score"],
                        -x["cost"],
                        x["duration"],
                        len(json.loads(x["tags"]))
                    )
                )
                conn.commit()
                return "\n".join(f"{t['id']}: {t['description']} (Cost: ${t['cost']}, Progress: {t['progress']}%, Time: {t['time_block'] or 'Unscheduled'})" 
                                for t in sorted_tasks)
            except sqlite3.Error as e:
                logging.error(f"Optimization error: {e}")
                return "Optimization failed."
            finally:
                conn.close()

    def check_dependencies(self, task_id):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                task = conn.execute("SELECT dependencies FROM tasks WHERE id = ?", (task_id,)).fetchone()
                if not task:
                    return True
                deps = json.loads(task[0])
                for dep in deps:
                    dep_task = conn.execute("SELECT completed FROM tasks WHERE description = ?", (dep,)).fetchone()
                    if dep_task and not dep_task[0]:
                        return False
                return True
            except sqlite3.Error as e:
                logging.error(f"Dependency check error for {task_id}: {e}")
                return True
            finally:
                conn.close()

    def list_tasks(self, username, filter_by="all"):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                query = "SELECT * FROM tasks WHERE assigned_to = ?"
                if filter_by == "pending":
                    query += " AND completed = 0"
                elif filter_by == "completed":
                    query += " AND completed = 1"
                tasks = conn.execute(query, (username,)).fetchall()
                if not tasks:
                    return "No tasks scheduled."
                output = f"Your {filter_by} Tasks:\n"
                for task in tasks:
                    task_dict = dict(zip([d[0] for d in conn.execute("PRAGMA table_info(tasks)").fetchall()], task))
                    status = "✔" if task_dict["completed"] else f"{task_dict['progress']}%"
                    tags = " ".join(json.loads(task_dict["tags"]))
                    output += (f"{task_dict['id']}: {task_dict['description']} | Deadline: {task_dict['deadline'] or 'None'} | "
                              f"Priority: {task_dict['priority']} | Cost: ${task_dict['cost']} | Tags: {tags} | {status} | "
                              f"Time: {task_dict['time_block'] or 'Unscheduled'}\n")
                    if task_dict["notes"]:
                        output += f"  Notes: {task_dict['notes']}\n"
                return output
            except sqlite3.Error as e:
                logging.error(f"List tasks error: {e}")
                return "Error listing tasks."
            finally:
                conn.close()

    def analyze_tasks(self, username):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                df = pd.read_sql_query("SELECT * FROM tasks WHERE assigned_to = ?", conn, params=(username,))
                if df.empty:
                    return "No tasks to analyze."
                completion_rate = (df["completed"] == 1).mean() * 100
                avg_cost = df["cost"].mean()
                avg_progress = df["progress"].mean()
                overdue = len(df[(df["deadline"] != "") & (pd.to_datetime(df["deadline"]) < datetime.datetime.now()) & (df["completed"] == 0)])
                category_breakdown = df["category"].value_counts().to_dict()
                busiest_day = pd.to_datetime(df["deadline"]).dt.day_name().mode()[0] if not df["deadline"].empty else "N/A"
                heatmap = df.groupby(pd.to_datetime(df["deadline"]).dt.hour)["id"].count().to_dict()
                return (f"Task Insights:\nCompletion Rate: {completion_rate:.2f}%\nAvg Cost: ${avg_cost:.2f}\n"
                        f"Avg Progress: {avg_progress:.2f}%\nOverdue Tasks: {overdue}\n"
                        f"Category Breakdown: {category_breakdown}\nBusiest Day: {busiest_day}\n"
                        f"Task Heatmap (Tasks per Hour): {heatmap}")
            except Exception as e:
                logging.error(f"Analysis error: {e}")
                return "Error generating insights."
            finally:
                conn.close()

    def update_progress(self, task_id, progress):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                if not conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone():
                    return "Task not found."
                if not 0 <= progress <= 100:
                    return "Progress must be between 0 and 100."
                conn.execute("UPDATE tasks SET progress = ? WHERE id = ?", (progress, task_id))
                if progress == 100:
                    return self.complete_task(task_id)
                conn.commit()
                return f"Progress updated for {task_id} to {progress}%."
            except sqlite3.Error as e:
                logging.error(f"Progress update error for {task_id}: {e}")
                return "Error updating progress."
            finally:
                conn.close()

    def cluster_tasks(self, username):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                df = pd.read_sql_query("SELECT id, description, duration, cost FROM tasks WHERE assigned_to = ? AND completed = 0", 
                                      conn, params=(username,))
                if len(df) < 3:
                    return
                X = np.array([[len(row["description"].split()), row["duration"], row["cost"]] for _, row in df.iterrows()])
                kmeans = KMeans(n_clusters=min(3, len(df)), random_state=42)
                clusters = kmeans.fit_predict(X)
                for (task_id, _), cluster in zip(df.iterrows(), clusters):
                    conn.execute("UPDATE tasks SET cluster = ? WHERE id = ?", (int(cluster), task_id))
                conn.commit()
                logging.info(f"Tasks clustered for {username}.")
            except Exception as e:
                logging.error(f"Clustering error: {e}")
            finally:
                conn.close()

    def assign_time_block(self, task):
        try:
            if task["completed"] or task["time_block"]:
                return
            start_hour, end_hour = self.preferences["work_hours"]
            duration = task["duration"] / 60
            available_slots = []
            current_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
            with db_lock:
                conn = sqlite3.connect(TASK_DB)
                for hour in range(start_hour, end_hour):
                    slot_start = current_time.replace(hour=hour)
                    slot_end = slot_start + datetime.timedelta(hours=1)
                    if slot_end <= current_time:
                        continue
                    slot_tasks = conn.execute("SELECT duration FROM tasks WHERE time_block = ?", (slot_start.isoformat(),)).fetchall()
                    if sum(t[0] / 60 for t in slot_tasks) + duration <= 1:
                        available_slots.append(slot_start)
                if available_slots:
                    task["time_block"] = random.choice(available_slots).isoformat()
                    conn.execute("UPDATE tasks SET time_block = ? WHERE id = ?", (task["time_block"], task["id"]))
                    conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Time block assignment error for {task['id']}: {e}")
        finally:
            conn.close()

    def proactive_reschedule(self, username):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                tasks = conn.execute("SELECT * FROM tasks WHERE assigned_to = ? AND completed = 0", (username,)).fetchall()
                current_time = datetime.datetime.now()
                for task in tasks:
                    task_dict = dict(zip([d[0] for d in conn.execute("PRAGMA table_info(tasks)").fetchall()], task))
                    if task_dict["deadline"] and current_time > datetime.datetime.fromisoformat(task_dict["deadline"]):
                        new_deadline = (current_time + datetime.timedelta(days=1)).isoformat()
                        conn.execute("UPDATE tasks SET deadline = ?, time_block = NULL WHERE id = ?", 
                                    (new_deadline, task_dict["id"]))
                        self.assign_time_block(task_dict)
                conn.commit()
                return "Overdue tasks rescheduled."
            except sqlite3.Error as e:
                logging.error(f"Reschedule error: {e}")
                return "Error rescheduling tasks."
            finally:
                conn.close()

    def predict_failure(self, username):
        with db_lock:
            try:
                conn = sqlite3.connect(TASK_DB)
                df = pd.read_sql_query("SELECT * FROM tasks WHERE assigned_to = ? AND completed = 0", conn, params=(username,))
                if df.empty:
                    return "No tasks to predict."
                risks = []
                for _, row in df.iterrows():
                    score = self.predict_priority(dict(row))
                    if score < 0.3 and datetime.datetime.fromisoformat(row["deadline"]) < datetime.datetime.now() + datetime.timedelta(days=1):
                        risks.append(row["id"])
                return f"Potential Failures: {', '.join(risks) if risks else 'None'}"
            except Exception as e:
                logging.error(f"Failure prediction error: {e}")
                return "Error predicting failures."
            finally:
                conn.close()

    def process_offline_queue(self):
        while True:
            with db_lock:
                try:
                    if self.offline_queue:
                        action, data = self.offline_queue.popleft()
                        if action == "add_task":
                            username = data.split('_')[0] if '_' in data else "default_user"
                            self.add_task(data, username)
                        logging.info(f"Processed offline task: {action} - {data}")
                    time.sleep(5)
                except Exception as e:
                    logging.error(f"Error processing offline queue: {e}")
                    time.sleep(5)

    def get_template(self, template_name):
        templates = {
            "default": {
                "description": "{}",
                "priority": "medium",
                "category": "general",
                "reminders": [300]
            }
        }
        return templates.get(template_name, templates["default"])

try:
    scheduler = TaskSchedulerAI()
except Exception as e:
    logging.error(f"Failed to create TaskSchedulerAI instance: {e}")
    raise

# --- Flask API ---
app = Flask(__name__)

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or token != f"Bearer {SECRET_KEY}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username, password = data.get("username"), data.get("password")
    success, message = security.register(username, password)
    return jsonify({"success": success, "message": message})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username, password = data.get("username"), data.get("password")
    success, message = security.login(username, password)
    if success:
        return jsonify({"success": True, "token": SECRET_KEY})
    return jsonify({"success": False, "message": message})

@app.route("/tasks", methods=["POST"])
@require_auth
def add_task_api():
    data = request.json
    username = data.get("username")
    task_input = data.get("task")
    template = data.get("template")
    if not username or not task_input:
        return jsonify({"message": "Username and task are required"}), 400
    result = scheduler.add_task(task_input, username, template)
    return jsonify({"message": result})

@app.route("/tasks", methods=["GET"])
@require_auth
def list_tasks_api():
    username = request.args.get("username")
    filter_by = request.args.get("filter", "all")
    if not username:
        return jsonify({"message": "Username is required"}), 400
    result = scheduler.list_tasks(username, filter_by)
    return jsonify({"tasks": result})

@app.route("/tasks/<task_id>/complete", methods=["PUT"])
@require_auth
def complete_task_api(task_id):
    result = scheduler.complete_task(task_id)
    return jsonify({"message": result})

@app.route("/optimize", methods=["GET"])
@require_auth
def optimize_api():
    result = scheduler.optimize_tasks()
    return jsonify({"schedule": result})

@app.route("/analyze", methods=["GET"])
@require_auth
def analyze_api():
    username = request.args.get("username")
    if not username:
        return jsonify({"message": "Username is required"}), 400
    result = scheduler.analyze_tasks(username)
    return jsonify({"insights": result})

@app.route("/reschedule", methods=["POST"])
@require_auth
def reschedule_api():
    data = request.json
    username = data.get("username")
    if not username:
        return jsonify({"message": "Username is required"}), 400
    result = scheduler.proactive_reschedule(username)
    return jsonify({"message": result})

@app.route("/predict", methods=["GET"])
@require_auth
def predict_api():
    username = request.args.get("username")
    if not username:
        return jsonify({"message": "Username is required"}), 400
    result = scheduler.predict_failure(username)
    return jsonify({"prediction": result})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)
