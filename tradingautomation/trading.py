from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.storage.jsonstore import JsonStore
from kivy.metrics import dp
from kivy.properties import ColorProperty, ListProperty, NumericProperty
from kivy.graphics import Color, Rectangle, RoundedRectangle, Line
from kivy.core.window import Window
import requests
import time
from datetime import datetime
import numpy as np
import os

Window.fullscreen = 'auto'
Window.clearcolor = (0.1, 0.1, 0.2, 1)

BYBIT_API_KEY = "eCme8ZIf5ishlOd3o9" 
BYBIT_BASE_URL = "https://api.bybit.com"

VIRTUZENAI_API_KEY = "AIzaSyBeMGH5FDILyFBiFkiDpIX1srFaZ5ELR8M"
VIRTUZENAI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

class TradingAccount:
    
    def __init__(self, usdt_balance=4308, btc_balance=0):
        self.usdt_balance = usdt_balance
        self.btc_balance = btc_balance

    def update_balances(self, usdt_balance, btc_balance):
        try:
            self.usdt_balance = usdt_balance
            self.btc_balance = btc_balance
            return f"USDT: ${self.usdt_balance:.2f}\nBTC: {self.btc_balance:.6f}"
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Balance update error - {str(e)}\n")
            return "USDT: $0.00\nBTC: 0.000000"

def get_candlestick_data(symbol="BTCUSDT", interval="60", limit=20, retries=3):
    
    endpoint = "/v5/market/kline"
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": limit}
    for attempt in range(retries):
        try:
            response = requests.get(f"{BYBIT_BASE_URL}{endpoint}", params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") == 0:
                candles = [
                    {
                        "timestamp": int(candle[0]) / 1000,
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    }
                    for candle in data.get("result", {}).get("list", [])
                ]
                if candles:
                    return candles
            error_msg = data.get("retMsg", "Unknown error")
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Bybit API error - {error_msg}\n")
            return None
        except requests.RequestException as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Bybit request error - {str(e)}\n")
            time.sleep(1)
    return None

def calculate_technical_indicators(candles):
   
    try:
        if not candles or len(candles) < 1:
            return {"sma": [], "ema": [], "rsi": None, "macd": None, "signal": None}

        closes = [c["close"] for c in candles]
        
        sma = []
        for i in range(len(closes)):
            if i >= 19:
                sma.append(sum(closes[i-19:i+1]) / 20)
            else:
                sma.append(None)
        
        ema = []
        k = 2 / (12 + 1)
        for i in range(len(closes)):
            if i == 0:
                ema.append(closes[0])
            else:
                ema.append((closes[i] * k) + (ema[i-1] * (1 - k)))
        
        rsi = None
        if len(closes) >= 15:
            gains = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
            losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi = 100 - (100 / (1 + rs))
        
        macd = signal = None
        if len(closes) >= 26:
            ema12 = closes[-1]
            k12 = 2 / (12 + 1)
            for price in closes[-12:-1][::-1]:
                ema12 = (price * k12) + (ema12 * (1 - k12))
            ema26 = closes[-1]
            k26 = 2 / (26 + 1)
            for price in closes[-26:-1][::-1]:
                ema26 = (price * k26) + (ema26 * (1 - k26))
            macd = ema12 - ema26
            macd_values = []
            for i in range(-9, 0):
                ema12_i = closes[i]
                for price in closes[i-12:i][::-1]:
                    ema12_i = (price * k12) + (ema12_i * (1 - k12))
                ema26_i = closes[i]
                for price in closes[i-26:i][::-1]:
                    ema26_i = (price * k26) + (ema26_i * (1 - k26))
                macd_values.append(ema12_i - ema26_i)
            signal = sum(macd_values) / len(macd_values) if macd_values else 0
        
        return {"sma": sma, "ema": ema, "rsi": rsi, "macd": macd, "signal": signal}
    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: Indicator calculation error - {str(e)}\n")
        return {"sma": [], "ema": [], "rsi": None, "macd": None, "signal": None}

def get_fundamental_data():
    return {
        "hashrate": 200 * 10**18,
        "tx_volume": 500000,
        "active_addresses": 900000
    }

def get_sentiment_analysis():
    return {
        "sentiment_score": 0.7,
        "social_media_volume": 10000,
        "news_tone": "Positive"
    }

def get_macroeconomic_data():
    return {
        "usd_interest_rate": 3.0,
        "inflation_rate": 2.0,
        "regulatory_news": "Neutral"
    }

def get_historical_patterns(candles):
    try:
        closes = [c["close"] for c in candles]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = np.std(returns) * np.sqrt(365) * 100 if returns else 0
        return {
            "avg_return": sum(returns) / len(returns) if returns else 0,
            "volatility": volatility
        }
    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: Historical patterns error - {str(e)}\n")
        return {"avg_return": 0, "volatility": 0}

def calculate_risk_metrics(candles, balance):
    try:
        closes = [c["close"] for c in candles]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        var_95 = np.percentile(returns, 5) * balance if returns else 0
        return {
            "var_95": var_95,
            "max_position": balance * 0.02
        }
    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: Risk metrics error - {str(e)}\n")
        return {"var_95": 0, "max_position": 0}

def get_alternative_data():
    return {
        "google_trends_score": 80,
        "miner_outflows": 100,
        "institutional_activity": "Moderate"
    }

def analyze_with_virtuzenai(candles, indicators, fundamentals, sentiment, macro, historical, risk, alternative, balance, current_price):
    try:
        candle_summary = "\n".join([
            f"Candle {i+1}: Open: ${c['open']:.2f}, High: ${c['high']:.2f}, Low: ${c['low']:.2f}, Close: ${c['close']:.2f}, Volume: {c['volume']:.2f}"
            for i, c in enumerate(candles[-5:])
        ])
        sma_str = "N/A" if not indicators['sma'] or not indicators['sma'][-1] else f"${indicators['sma'][-1]:.2f}"
        ema_str = "N/A" if not indicators['ema'] else f"${indicators['ema'][-1]:.2f}"
        rsi_str = "N/A" if indicators['rsi'] is None else f"{indicators['rsi']:.2f}"
        macd_str = "N/A" if indicators['macd'] is None else f"{indicators['macd']:.2f}"
        signal_str = "N/A" if indicators['signal'] is None else f"{indicators['signal']:.2f}"
        indicator_summary = f"SMA: {sma_str}, EMA: {ema_str}, RSI: {rsi_str}, MACD: {macd_str}, Signal: {signal_str}"
        fundamental_summary = (
            f"Hashrate: {fundamentals['hashrate'] / 10**18:.2f} EH/s, "
            f"Tx Volume: {fundamentals['tx_volume']:,}, Active Addresses: {fundamentals['active_addresses']:,}"
        )
        sentiment_summary = (
            f"Sentiment Score: {sentiment['sentiment_score']:.2f}, "
            f"Social Media Volume: {sentiment['social_media_volume']:,}, News Tone: {sentiment['news_tone']}"
        )
        macro_summary = (
            f"USD Interest Rate: {macro['usd_interest_rate']}%, Inflation: {macro['inflation_rate']}%, "
            f"Regulatory News: {macro['regulatory_news']}"
        )
        historical_summary = (
            f"Avg Return: {historical['avg_return']:.4f}, Volatility: {historical['volatility']:.2f}%"
        )
        risk_summary = (
            f"VaR (95%): ${risk['var_95']:.2f}, Max Position: ${risk['max_position']:.2f}"
        )
        alternative_summary = (
            f"Google Trends: {alternative['google_trends_score']}, Miner Outflows: {alternative['miner_outflows']}, "
            f"Institutional Activity: {alternative['institutional_activity']}"
        )
        balance_summary = f"Your balance: ${balance.usdt_balance:.2f}, BTC: {balance.btc_balance:.6f}"

        prompt = (
        f"Please act as an expert and smartest Crypto Trader and also make your tone more human and analyze the BTCUSDT spot market and provide a trading suggestion ('buy', 'sell', or 'hold') with a 1 sentence explanation and compare the price from previous BTC price to updated BTC price:\n"
            f"Current Price: ${current_price:.2f}\n"
            f"Recent Candles:\n{candle_summary}\n"
            f"Technical Indicators:\n{indicator_summary}\n"
            f"Fundamentals:\n{fundamental_summary}\n"
            f"Sentiment:\n{sentiment_summary}\n"
            f"Macroeconomic Factors:\n{macro_summary}\n"
            f"Historical Patterns:\n{historical_summary}\n"
            f"Risk Metrics:\n{risk_summary}\n"
            f"Alternative Data:\n{alternative_summary}\n"
            f"Balance:\n{balance_summary}\n"
            f"Format your response as: 'suggestion: explanation.'"
        )

        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(
            f"{VIRTUZENAI_API_URL}?key={VIRTUZENAI_API_KEY}",
            headers=headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        if ":" in text:
            action, explanation = text.split(":", 1)
            action = action.strip().lower()
            explanation = explanation.strip()
            explanation = explanation.replace("RSI", "[color=33ccff]RSI[/color]")
            explanation = explanation.replace("MACD", "[color=33ccff]MACD[/color]")
            explanation = explanation.replace("volatility", "[color=33ccff]volatility[/color]")
            if action == "buy":
                explanation += f" [color=33ff33]Buying[/color] is recommended."
            elif action == "sell":
                explanation += f" [color=ff3333]Selling[/color] is advised."
            else:
                explanation += f" [color=ffff33]Holding[/color] is the best course."
            return action, explanation
        return "hold", "Invalid response format from VirtuzenAi. [color=ffff33]Holding[/color] is recommended."
    except requests.RequestException as e:
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: Gemini API error - {str(e)}\n")
        return "hold", f"Failed to get VirtuzenAi analysis due to API error: {str(e)}. [color=ffff33]Holding[/color] is recommended."

class GradientBackground(Widget):
    background_color = ColorProperty([0.1, 0.1, 0.2, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(*self.background_color)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

class FuturisticButton(Button):
    button_color = ColorProperty([0.2, 0.6, 1, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_color = [0, 0, 0, 0]
        self.font_size = dp(16)
        with self.canvas.before:
            Color(*self.button_color)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(15)])
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

class CandlestickChart(Widget):
    candles = ListProperty([])
    sma = ListProperty([])
    ema = ListProperty([])
    zoom_level = NumericProperty(1.0) 
    pan_offset = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(candles=self.update_chart, sma=self.update_chart, ema=self.update_chart,
                  pos=self.update_chart, size=self.update_chart, zoom_level=self.update_chart,
                  pan_offset=self.update_chart)

    def update_chart(self, *args):
        try:
            self.canvas.clear()
            with self.canvas:
                if not self.candles:
                    return

                highs = [c["high"] for c in self.candles]
                lows = [c["low"] for c in self.candles]
                volumes = [c["volume"] for c in self.candles]
                if not highs or not lows or not volumes:
                    return
                max_price = max(highs)
                min_price = min(lows)
                price_range = max_price - min_price
                max_volume = max(volumes)
                if price_range == 0 or max_volume == 0:
                    return
                chart_width = self.width - dp(70) 
                chart_height = self.height - dp(70)
                volume_height = dp(40)
                num_candles = len(self.candles)
                visible_candles = int(20 / self.zoom_level) 
                candle_width = (chart_width / visible_candles) * 0.7 * self.zoom_level
                spacing = (chart_width / visible_candles) * 0.3 / self.zoom_level

                self.pan_offset = max(min(self.pan_offset, num_candles - visible_candles), 0)
                start_idx = int(self.pan_offset)
                end_idx = min(start_idx + visible_candles, num_candles)

                Color(0.2, 0.2, 0.3, 0.5)
                num_price_levels = 6
                for i in range(num_price_levels):
                    y = self.y + volume_height + (i / (num_price_levels - 1)) * chart_height
                    Line(points=[self.x, y, self.x + chart_width, y], width=0.5, dash_length=5, dash_offset=5)
                for i in range(start_idx, end_idx, 5): 
                    x = self.x + (i - start_idx) * (candle_width + spacing) + spacing / 2
                    Line(points=[x, self.y, x, self.y + volume_height + chart_height], width=0.5, dash_length=5, dash_offset=5)

                Color(0.6, 0.6, 0.8, 1)
                for i in range(num_price_levels):
                    price = min_price + (price_range * i / (num_price_levels - 1))
                    y = self.y + volume_height + (i / (num_price_levels - 1)) * chart_height
                    Line(points=[self.x + chart_width, y, self.x + chart_width + dp(5), y], width=1)
                    Rectangle(
                        pos=(self.x + chart_width + dp(10), y - dp(5)),
                        size=(dp(60), dp(10)),
                        texture=Label(text=f"${price:.0f}", font_size=dp(10), color=(0.6, 0.6, 0.8, 1)).texture
                    )

                for i in range(start_idx, end_idx, 5):
                    if i < len(self.candles):
                        timestamp = self.candles[i]["timestamp"]
                        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M")
                        x = self.x + (i - start_idx) * (candle_width + spacing) + spacing / 2
                        Rectangle(
                            pos=(x - dp(20), self.y + volume_height - dp(15)),
                            size=(dp(40), dp(10)),
                            texture=Label(text=time_str, font_size=dp(10), color=(0.6, 0.6, 0.8, 1)).texture
                        )

                for i in range(start_idx, end_idx):
                    candle = self.candles[i]
                    volume = candle["volume"]
                    volume_scaled = (volume / max_volume) * volume_height
                    x = self.x + (i - start_idx) * (candle_width + spacing) + spacing / 2
                    if candle["close"] >= candle["open"]:
                        Color(0, 0.8, 0, 0.8)
                    else:
                        Color(0.8, 0, 0, 0.8) 
                    Rectangle(pos=(x, self.y), size=(candle_width, volume_scaled))

                for i in range(start_idx, end_idx):
                    candle = self.candles[i]
                    open_price = candle["open"]
                    close_price = candle["close"]
                    high_price = candle["high"]
                    low_price = candle["low"]

                    open_y = self.y + volume_height + ((open_price - min_price) / price_range) * chart_height
                    close_y = self.y + volume_height + ((close_price - min_price) / price_range) * chart_height
                    high_y = self.y + volume_height + ((high_price - min_price) / price_range) * chart_height
                    low_y = self.y + volume_height + ((low_price - min_price) / price_range) * chart_height

                    x = self.x + (i - start_idx) * (candle_width + spacing) + spacing / 2

                    Color(0.8, 0.8, 0.8, 1)
                    Line(points=[x + candle_width / 2, low_y, x + candle_width / 2, high_y], width=1.5)

                    if close_price >= open_price:
                        Color(0, 1, 0, 1)
                        Rectangle(pos=(x, open_y), size=(candle_width, close_y - open_y))
                    else:
                        Color(1, 0, 0, 1)
                        Rectangle(pos=(x, close_y), size=(candle_width, open_y - close_y))
                points = []
                for i in range(start_idx, end_idx):
                    sma_value = self.sma[i] if self.sma else None
                    if sma_value is not None:
                        x = self.x + (i - start_idx) * (candle_width + spacing) + spacing / 2 + candle_width / 2
                        y = self.y + volume_height + ((sma_value - min_price) / price_range) * chart_height
                        points.extend([x, y])
                if len(points) >= 4:
                    Color(1, 1, 0, 1)
                    Line(points=points, width=1.5)
                    last_x, last_y = points[-2], points[-1]
                    Rectangle(
                        pos=(last_x + dp(10), last_y - dp(5)),
                        size=(dp(50), dp(10)),
                        texture=Label(text="SMA 20", font_size=dp(10), color=(1, 1, 0, 1)).texture
                    )
                points = []
                for i in range(start_idx, end_idx):
                    ema_value = self.ema[i] if self.ema else None
                    if ema_value is not None:
                        x = self.x + (i - start_idx) * (candle_width + spacing) + spacing / 2 + candle_width / 2
                        y = self.y + volume_height + ((ema_value - min_price) / price_range) * chart_height
                        points.extend([x, y])
                if len(points) >= 4:
                    Color(0, 1, 1, 1)
                    Line(points=points, width=1.5)
                    last_x, last_y = points[-2], points[-1]
                    Rectangle(
                        pos=(last_x + dp(10), last_y - dp(15)),
                        size=(dp(50), dp(10)),
                        texture=Label(text="EMA 12", font_size=dp(10), color=(0, 1, 1, 1)).texture
                    )
                current_price = self.candles[-1]["close"]
                price_y = self.y + volume_height + ((current_price - min_price) / price_range) * chart_height
                Color(0.2, 0.8, 1, 1)
                Line(points=[self.x, price_y, self.x + chart_width, price_y], width=1, dash_length=5, dash_offset=5)
                Rectangle(
                    pos=(self.x + chart_width + dp(10), price_y - dp(5)),
                    size=(dp(60), dp(10)),
                    texture=Label(text=f"${current_price:.0f}", font_size=dp(10), color=(0.2, 0.8, 1, 1)).texture
                )
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Chart update error - {str(e)}\n")

class TradingApp(App):
    def __init__(self):
        super().__init__()
        self.account = TradingAccount()
        self.store = JsonStore('trading_config.json')
        self.running = False
        self.prices = []
        self.actions = []
        self.update_event = None
        self.trade_history = []
        self.last_candles = None
        self.loading_texts = ["Fetching data...", "Analyzing market...", "Updating signals..."]
        self.loading_index = 0
        self.loading_opacity = 1.0
        self.loading_direction = -0.05

    def build(self):
        self.layout = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(10))
        self.background = GradientBackground()
        self.layout.add_widget(self.background, index=len(self.layout.children))
        title = Label(
            text="[b]Crypto Trading Automation[/b]",
            markup=True,
            font_size=dp(24),
            color=[0.2, 0.8, 1, 1],
            size_hint=(1, 0.1)
        )
        self.layout.add_widget(title)
        chart_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.4))
        self.chart = CandlestickChart()
        chart_layout.add_widget(self.chart)
        nav_layout = BoxLayout(size_hint=(1, 0.1), spacing=dp(10))
        zoom_in_btn = FuturisticButton(text='+', size_hint=(0.2, 1))
        zoom_in_btn.bind(on_press=self.zoom_in)
        nav_layout.add_widget(zoom_in_btn)
        zoom_out_btn = FuturisticButton(text='-', size_hint=(0.2, 1))
        zoom_out_btn.bind(on_press=self.zoom_out)
        nav_layout.add_widget(zoom_out_btn)
        pan_left_btn = FuturisticButton(text='<', size_hint=(0.2, 1))
        pan_left_btn.bind(on_press=self.pan_left)
        nav_layout.add_widget(pan_left_btn)
        pan_right_btn = FuturisticButton(text='>', size_hint=(0.2, 1))
        pan_right_btn.bind(on_press=self.pan_right)
        nav_layout.add_widget(pan_right_btn)
        chart_layout.add_widget(nav_layout)
        self.layout.add_widget(chart_layout)
        self.scroll = ScrollView() 
        content = GridLayout(cols=1, spacing=dp(10), size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        self.loading_label = Label(
            text='',
            size_hint=(1, None),
            height=dp(35),
            color=[0.8, 0.8, 1, 1],
            font_size=dp(16),
            markup=True
        )
        content.add_widget(self.loading_label)
        self.status_label = Label(
            text='Status: Stopped',
            size_hint=(1, None),
            height=dp(35),
            color=[0.8, 0.8, 1, 1],
            font_size=dp(16)
        )
        content.add_widget(self.status_label)
        self.price_label = Label(
            text='BTC Price: N/A',
            size_hint=(1, None),
            height=dp(40),
            color=[0.2, 1, 0.6, 1],
            font_size=dp(18)
        )
        content.add_widget(self.price_label)
        self.balance_label = Label(
            text=self.account.update_balances(self.account.usdt_balance, self.account.btc_balance),
            size_hint=(1, None),
            height=dp(55),
            color=[1, 1, 1, 1],
            font_size=dp(16)
        )
        content.add_widget(self.balance_label)
        self.action_label = Label(
            text='VirtuzenAi: Awaiting data...',
            size_hint=(1, None),
            height=dp(150),
            color=[1, 1, 1, 1],
            font_size=dp(14),
            text_size=(dp(300), None),
            markup=True
        )
        content.add_widget(self.action_label)
        self.history_label = Label(
            text='Trade History: None',
            size_hint=(1, None),
            height=dp(90),
            color=[0.6, 0.6, 0.8, 1],
            font_size=dp(12),
            text_size=(dp(300), None)
        )
        content.add_widget(self.history_label)

        self.scroll.add_widget(content)
        self.layout.add_widget(self.scroll)
        button_layout = BoxLayout(size_hint=(1, 0.15), spacing=dp(15))
        self.start_btn = FuturisticButton(text='Start', size_hint=(0.25, 1))
        self.start_btn.bind(on_press=self.start_trading)
        button_layout.add_widget(self.start_btn)

        self.stop_btn = FuturisticButton(text='Stop', size_hint=(0.25, 1), disabled=True)
        self.stop_btn.bind(on_press=self.stop_trading)
        button_layout.add_widget(self.stop_btn)

        refresh_btn = FuturisticButton(text='Refresh', size_hint=(0.25, 1))
        refresh_btn.bind(on_press=self.refresh_data)
        button_layout.add_widget(refresh_btn)
        explanation_btn = FuturisticButton(text='Exp', size_hint=(0.25, 1))
        explanation_btn.bind(on_press=self.show_explanation)
        button_layout.add_widget(explanation_btn)

        self.layout.add_widget(button_layout)
        Clock.schedule_interval(self.update_loading_animation, 0.05)

        return self.layout

    def show_explanation(self, instance):
        try:
            self.scroll.scroll_to(self.action_label, padding=dp(10), animate=True)
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Show explanation error - {str(e)}\n")

    def update_loading_animation(self, dt):
        try:
            if self.loading_label.text:
                self.loading_opacity += self.loading_direction
                if self.loading_opacity <= 0.3 or self.loading_opacity >= 1.0:
                    self.loading_direction *= -1
                self.loading_label.color = [0.8, 0.8, 1, self.loading_opacity]
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Loading animation error - {str(e)}\n")

    def zoom_in(self, instance):
        try:
            self.chart.zoom_level = min(self.chart.zoom_level + 0.2, 2.0)
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Zoom in error - {str(e)}\n")

    def zoom_out(self, instance):
        try:
            self.chart.zoom_level = max(self.chart.zoom_level - 0.2, 0.5)
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Zoom out error - {str(e)}\n")

    def pan_left(self, instance):
        try:
            self.chart.pan_offset += 1
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Pan left error - {str(e)}\n")

    def pan_right(self, instance):
        try:
            self.chart.pan_offset -= 1
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Pan right error - {str(e)}\n")

    def refresh_data(self, instance):
        try:
            if self.running:
                self.update_ui(0)
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Refresh error - {str(e)}\n")

    def update_ui(self, dt):
        try:
            if not self.running and dt != 0:  # Allow manual refresh
                return

            # Show loading indicator
            self.loading_label.text = f"[i]{self.loading_texts[self.loading_index]}[/i]"
            self.loading_index = (self.loading_index + 1) % len(self.loading_texts)
            self.loading_opacity = 1.0
            self.loading_direction = -0.05

            candles = get_candlestick_data()
            if not candles:
                self.loading_label.text = ''
                self.status_label.text = 'Status: Failed to fetch data'
                self.action_label.text = 'VirtuzenAi: Unable to fetch market data'
                popup = Popup(
                    title='Error',
                    content=Label(text='Check internet or try again'),
                    size_hint=(0.5, 0.5)
                )
                popup.open()
                return

            self.last_candles = candles
            current_price = candles[0]["close"]
            indicators = calculate_technical_indicators(candles)
            fundamentals = get_fundamental_data()
            sentiment = get_sentiment_analysis()
            macro = get_macroeconomic_data()
            historical = get_historical_patterns(candles)
            risk = calculate_risk_metrics(candles, self.account.usdt_balance)
            alternative = get_alternative_data()

            # Update candlestick chart
            self.chart.candles = candles
            self.chart.sma = indicators["sma"]
            self.chart.ema = indicators["ema"]

            # Get VirtuzenAi analysis
            action, explanation = analyze_with_virtuzenai(
                candles, indicators, fundamentals, sentiment, macro, historical,
                risk, alternative, self.account, current_price
            )

            self.prices.append(current_price)
            self.actions.append(action)
            if len(self.prices) > 20:
                self.prices.pop(0)
                self.actions.pop(0)

            # Update UI and hide loading
            self.loading_label.text = ''
            self.price_label.text = f'BTC Price: ${current_price:.2f}'
            self.status_label.text = f'Status: Active ({datetime.now().strftime("%H:%M:%S")})'
            self.action_label.text = f"VirtuzenAi suggests: [b]{action.upper()}[/b]\nVirtuzenAi: {explanation}"
            self.balance_label.text = self.account.update_balances(self.account.usdt_balance, self.account.btc_balance)

            # Simulate trades
            trade_msg = ""
            if action == "buy" and self.account.usdt_balance >= 215:
                btc_qty = 0.0025
                cost = btc_qty * current_price
                self.account.update_balances(self.account.usdt_balance - cost, self.account.btc_balance + btc_qty)
                trade_msg = f"Bought {btc_qty:.6f} BTC at ${current_price:.2f}"
            elif action == "sell" and self.account.btc_balance >= 0.0025:
                btc_qty = 0.0025
                proceeds = btc_qty * current_price
                self.account.update_balances(self.account.usdt_balance + proceeds, self.account.btc_balance - btc_qty)
                trade_msg = f"Sold {btc_qty:.6f} BTC at ${current_price:.2f}"

            # Update history
            if trade_msg or action != "hold":
                self.trade_history.append(f"{datetime.now().strftime('%H:%M:%S')}: {action.upper()} - {trade_msg or explanation[:30] + '...'}")
                if len(self.trade_history) > 4:
                    self.trade_history.pop(0)
                self.history_label.text = "Trade History:\n" + "\n".join(self.trade_history)

            # Log to file
            try:
                with open('trades.txt', 'a') as f:
                    f.write(f"{datetime.now()},{current_price},{action},{explanation.replace('[color=33ccff]', '').replace('[/color]', '').replace('[color=33ff33]', '').replace('[color=ff3333]', '').replace('[color=ffff33]', '')}\n")
            except Exception as e:
                with open('error_log.txt', 'a') as f:
                    f.write(f"{datetime.now()}: Trade log error - {str(e)}\n")
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Update UI error - {str(e)}\n")
            self.loading_label.text = ''
            self.status_label.text = 'Status: Error occurred'
            popup = Popup(
                title='Error',
                content=Label(text='An error occurred. Check error_log.txt'),
                size_hint=(0.5, 0.5)
            )
            popup.open()

    def start_trading(self, instance):
        """Starts the trading loop."""
        try:
            self.running = True
            self.start_btn.disabled = True
            self.stop_btn.disabled = False
            self.update_event = Clock.schedule_interval(self.update_ui, 20)
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Start trading error - {str(e)}\n")
            popup = Popup(
                title='Error',
                content=Label(text='Failed to start trading'),
                size_hint=(0.5, 0.5)
            )
            popup.open()

    def stop_trading(self, instance):
        """Stops the trading loop."""
        try:
            self.running = False
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
            if self.update_event:
                self.update_event.cancel()
            self.status_label.text = 'Status: Stopped'
            self.action_label.text = 'VirtuzenAi: Awaiting signal...'
            self.loading_label.text = ''
            self.chart.candles = []
            self.chart.sma = []
            self.chart.ema = []
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                f.write(f"{datetime.now()}: Stop trading error - {str(e)}\n")

# --- Entry Point ---

if __name__ == '__main__':
    try:
        TradingApp().run()
    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: Main app error - {str(e)}\n")
