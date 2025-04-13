import requests
import time
import json
from datetime import datetime
import plotly.graph_objects as go
import os
import threading
from flask import Flask, send_from_directory, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

trading_data = {
    "prices": [],
    "times": [],
    "actions": [],
    "account": {"usd_balance": 1000.0, "btc_balance": 0.0},
    "latest": {}
}
max_points = 50

CMC_API_KEY = os.getenv("CMC_API_KEY")
CMC_BASE_URL = "https://pro-api.coinmarketcap.com"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

class TradingAccount:
    def __init__(self, usd_balance=1000, btc_balance=0):
        self.usd_balance = usd_balance
        self.btc_balance = btc_balance

    def buy_btc(self, btc_amount, price):
        cost = btc_amount * price
        if cost <= self.usd_balance:
            self.usd_balance -= cost
            self.btc_balance += btc_amount
            print(f"Bought {btc_amount:.6f} BTC at ${price:.2f}. USD: ${self.usd_balance:.2f}, BTC: {self.btc_balance:.6f}")
            return True
        else:
            print("Insufficient USD balance")
            return False

    def sell_btc(self, btc_amount, price):
        if btc_amount <= self.btc_balance:
            revenue = btc_amount * price
            self.usd_balance += revenue
            self.btc_balance -= btc_amount
            print(f"Sold {btc_amount:.6f} BTC at ${price:.2f}. USD: ${self.usd_balance:.2f}, BTC: {self.btc_balance:.6f}")
            return True
        else:
            print("Insufficient BTC balance")
            return False

def get_btc_price():
    if not CMC_API_KEY:
        print("Error: CMC_API_KEY not set")
        return None
    url = f"{CMC_BASE_URL}/v1/cryptocurrency/quotes/latest"
    headers = {"Accepts": "application/json", "X-CMC_PRO_API_KEY": CMC_API_KEY}
    params = {"symbol": "BTC", "convert": "USD"}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if "data" in data and "BTC" in data["data"]:
            price = data["data"]["BTC"]["quote"]["USD"]["price"]
            print(f"Fetched BTC price: ${price:.2f}")
            return price
        else:
            print(f"Error: BTC data not found - {json.dumps(data, indent=2)}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching price: {e}")
        return None

def analyze_with_gemini(price, prev_price):
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set")
        return None, "API key missing"
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"Current BTC price: ${price:.2f}. Previous price: ${prev_price:.2f}. "
        f"Analyze the price movement and suggest a trading action: 'buy', 'sell', or 'hold'. "
        f"Provide a one-word action followed by a brief explanation (1-2 sentences). "
        f"Example: 'buy: Price is rising steadily, indicating a potential uptrend.' "
        f"Return the response in this format."
    )
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        if ":" in text:
            action, explanation = text.split(":", 1)
            print(f"Gemini AI: {action.strip().lower()}: {explanation.strip()}")
            return action.strip().lower(), explanation.strip()
        else:
            print(f"Error: Invalid Gemini response - {text}")
            return None, "Invalid response format"
    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"Error with Gemini API: {e}")
        return None, str(e)

def save_plot(prices, times, actions):
    if not prices or len(prices) < 2:
        print("Not enough data to plot")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=prices, mode='lines', name='BTC Price', line=dict(color='blue')))
    buy_times, buy_prices = [], []
    sell_times, sell_prices = [], []
    for i, action in enumerate(actions):
        if action == "buy":
            buy_times.append(times[i])
            buy_prices.append(prices[i])
        elif action == "sell":
            sell_times.append(times[i])
            sell_prices.append(prices[i])
    if buy_times:
        fig.add_trace(go.Scatter(x=buy_times, y=buy_prices, mode='markers', name='Buy',
                                marker=dict(symbol='triangle-up', size=10, color='green')))
    if sell_times:
        fig.add_trace(go.Scatter(x=sell_times, y=sell_prices, mode='markers', name='Sell',
                                marker=dict(symbol='triangle-down', size=10, color='red')))
    fig.update_layout(
        title="BTC Price and Trading Signals",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        xaxis_tickangle=45,
        showlegend=True,
        template="plotly_white"
    )
    output_file = "static/btc_plot.html"
    try:
        fig.write_html(output_file)
        print(f"Plot saved to {output_file}")
    except Exception as e:
        print(f"Error saving plot: {e}")

def trading_bot():
    account = TradingAccount()
    prev_price = None
    csv_file = "static/trades.csv"
    try:
        with open(csv_file, 'w') as f:
            f.write("time,price,action,explanation\n")
        print("Initialized trades.csv")
    except Exception as e:
        print(f"Error initializing trades.csv: {e}")
    print("Starting trading bot...")
    while True:
        price = get_btc_price()
        if price:
            current_time = datetime.now()
            trading_data["prices"].append(price)
            trading_data["times"].append(current_time)
            if len(trading_data["prices"]) > max_points:
                trading_data["prices"].pop(0)
                trading_data["times"].pop(0)
                if trading_data["actions"]:
                    trading_data["actions"].pop(0)
            action, explanation = None, ""
            if prev_price:
                action, explanation = analyze_with_gemini(price, prev_price)
                if action:
                    trading_data["actions"].append(action)
                    if action == "buy":
                        account.buy_btc(0.01, price)
                    elif action == "sell":
                        account.sell_btc(0.01, price)
                    try:
                        with open(csv_file, 'a') as f:
                            f.write(f"{current_time},{price},\"{action}\",\"{explanation}\"\n")
                        print(f"Wrote trade to CSV: {current_time}, {price}, {action}")
                    except Exception as e:
                        print(f"Error writing to trades.csv: {e}")
            prev_price = price
            trading_data["account"] = {"usd_balance": account.usd_balance, "btc_balance": account.btc_balance}
            trading_data["latest"] = {
                "time": current_time.isoformat(),
                "price": price,
                "action": action,
                "explanation": explanation
            }
            if len(trading_data["prices"]) >= 2:
                save_plot(trading_data["prices"], trading_data["times"], trading_data["actions"])
        else:
            print("No price data, skipping cycle")
        time.sleep(30)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/data')
def get_data():
    print(f"Serving /api/data: {trading_data['latest']}")
    return jsonify(trading_data["latest"])

@app.route('/api/trades')
def get_trades():
    trades = []
    try:
        with open('static/trades.csv', 'r') as f:
            lines = f.readlines()
            headers = lines[0].strip().split(',')
            for line in lines[1:]:
                values = line.strip().split(',', 3)
                trades.append({
                    "time": values[0],
                    "price": float(values[1]),
                    "action": values[2].strip('"'),
                    "explanation": values[3].strip('"')
                })
        print(f"Serving /api/trades: {len(trades)} trades")
    except Exception as e:
        print(f"Error reading trades.csv: {e}")
    return jsonify(trades)

@app.route('/api/balances')
def get_balances():
    print(f"Serving /api/balances: {trading_data['account']}")
    return jsonify(trading_data["account"])

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    threading.Thread(target=trading_bot, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
