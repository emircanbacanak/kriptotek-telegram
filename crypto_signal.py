import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from binance.client import Client
import pandas as pd
import ta
from datetime import datetime, timedelta
import telegram
from urllib3.exceptions import InsecureRequestWarning
import urllib3
from decimal import Decimal, ROUND_DOWN, getcontext
import json
import aiohttp

# SSL uyarılarını kapat
urllib3.disable_warnings(InsecureRequestWarning)

# Telegram Bot ayarları
TELEGRAM_TOKEN = "7872345042:AAE6Om2LGtz1QjqfZz8ge0em6Gw29llzFno"
TELEGRAM_CHAT_ID = "847081095"

# ===== AYARLAR =====
MIN_VOLUME = 10000000
VOLUME_MA = 20
RSI_OVERSOLD, RSI_OVERBOUGHT = 40, 60
MFI_BULL, MFI_BEAR = 65, 35
MIN_STRENGTH, HIGH_STRENGTH = 60, 80
DAILY_LIMIT, WEEKLY_LIMIT = 8, 40
MIN_DAYS = 30
COOLDOWN = 2
HIGH_TARGET, HIGH_STOP = 1.03, 0.985
MED_TARGET, MED_STOP = 1.025, 0.99
LOW_TARGET, LOW_STOP = 1.02, 0.995
HIGH_LEV, MED_LEV, LOW_LEV = 15, 10, 5
STABLECOINS = ['USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'BUSDUSDT', 'USDPUSDT', 'USDTUSDT']

# Binance client oluştur (globalde)
client = Client()

# Telegram bot oluştur
bot = telegram.Bot(token=TELEGRAM_TOKEN)

async def send_telegram_message(message):
    """Telegram'a mesaj gönder"""
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')

# Dinamik fiyat formatlama fonksiyonu
def format_price(price, ref_price=None):
    """
    Fiyatı, referans fiyatın ondalık basamak sayısı kadar string olarak döndürür.
    float hassasiyeti olmadan, gereksiz yuvarlama veya fazla basamak olmadan gösterir.
    """
    if ref_price is not None:
        s = str(ref_price)
        if 'e' in s or 'E' in s:
            # Bilimsel gösterim varsa düzelt
            s = f"{ref_price:.20f}".rstrip('0').rstrip('.')
        if '.' in s:
            dec = len(s.split('.')[-1])
            # Decimal ile hassasiyetli kısaltma
            getcontext().prec = dec + 8
            d_price = Decimal(str(price)).quantize(Decimal('1.' + '0'*dec), rounding=ROUND_DOWN)
            return format(d_price, f'.{dec}f').rstrip('0').rstrip('.') if dec > 0 else str(int(d_price))
        else:
            return str(int(round(price)))
    else:
        # ref_price yoksa, eski davranış
        if price >= 1:
            return f"{price:.4f}".rstrip('0').rstrip('.')
        elif price >= 0.01:
            return f"{price:.6f}".rstrip('0').rstrip('.')
        elif price >= 0.0001:
            return f"{price:.8f}".rstrip('0').rstrip('.')
        else:
            return f"{price:.10f}".rstrip('0').rstrip('.')

def calculate_signal_strength(df, signals):
    """Sinyal güvenilirlik skorunu hesapla (0-100)"""
    strength = 0
    
    # 1. RSI gücü (0-20 puan)
    rsi = df['rsi'].iloc[-1]
    if signals['1h'] == 1:  # ALIŞ sinyali
        if rsi < 30: strength += 20
        elif rsi < RSI_OVERSOLD: strength += 15
        elif rsi < 50: strength += 10
    else:  # SATIŞ sinyali
        if rsi > 70: strength += 20
        elif rsi > RSI_OVERBOUGHT: strength += 15
        elif rsi > 50: strength += 10
    
    # 2. MACD gücü (0-20 puan)
    macd = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    if signals['1h'] == 1 and macd > macd_signal:
        strength += 20
    elif signals['1h'] == -1 and macd < macd_signal:
        strength += 20
    
    # 3. Trend gücü (0-20 puan)
    if df['trend_bullish'].iloc[-1] and signals['1h'] == 1:
        strength += 20
    elif df['trend_bearish'].iloc[-1] and signals['1h'] == -1:
        strength += 20
    
    # 4. Hacim gücü (0-20 puan)
    if df['enough_volume'].iloc[-1]:
        strength += 20
    
    # 5. MA gücü (0-20 puan)
    if df['ma_bullish'].iloc[-1] and signals['1h'] == 1:
        strength += 20
    elif df['ma_bearish'].iloc[-1] and signals['1h'] == -1:
        strength += 20
    
    return min(strength, 100)

def create_signal_message(symbol, price, signals, signal_strength=0):
    """Sinyal mesajını oluştur (AL/SAT başlıkta)"""
    price_str = format_price(price, price)  # Fiyatın kendi basamağı kadar
    signal_1h = "ALIŞ" if signals['1h'] == 1 else "SATIŞ"
    signal_2h = "ALIŞ" if signals['2h'] == 1 else "SATIŞ"
    signal_1d = "ALIŞ" if signals['1d'] == 1 else "SATIŞ"
    buy_count = sum(1 for s in signals.values() if s == 1)
    sell_count = sum(1 for s in signals.values() if s == -1)
    
    # Sinyal gücüne göre risk ayarlama
    if signal_strength >= HIGH_STRENGTH:
        risk_level = "🟢 DÜŞÜK RİSK"
        target_multiplier = HIGH_TARGET
        stop_multiplier = HIGH_STOP
        leverage_suggestion = f"{MED_LEV}x - {HIGH_LEV}x"
    elif signal_strength >= MIN_STRENGTH:
        risk_level = "🟡 ORTA RİSK"
        target_multiplier = MED_TARGET
        stop_multiplier = MED_STOP
        leverage_suggestion = f"{LOW_LEV}x - {MED_LEV}x"
    else:
        risk_level = "🔴 YÜKSEK RİSK"
        target_multiplier = LOW_TARGET
        stop_multiplier = LOW_STOP
        leverage_suggestion = f"3x - {LOW_LEV}x"
    
    if buy_count >= 2:
        dominant_signal = "ALIŞ"
        target_price = price * target_multiplier
        stop_loss = price * stop_multiplier
        sinyal_tipi = "AL SİNYALİ"
    elif sell_count >= 2:
        dominant_signal = "SATIŞ"
        target_price = price * (2 - target_multiplier)  # Ters hesaplama
        stop_loss = price * (2 - stop_multiplier)       # Ters hesaplama
        sinyal_tipi = "SAT SİNYALİ"
    else:
        return None, None, None, None, None
    
    # Hedef ve stop fiyatlarını, fiyatın ondalık basamağı kadar formatla
    target_price_str = format_price(target_price, price)
    stop_loss_str = format_price(stop_loss, price)
    
    # Güvenilirlik emoji
    if signal_strength >= HIGH_STRENGTH:
        confidence_emoji = "🔥"
    elif signal_strength >= MIN_STRENGTH:
        confidence_emoji = "⚡"
    else:
        confidence_emoji = "⚠️"
    
    message = f"""
{confidence_emoji} {sinyal_tipi} {confidence_emoji}

📊 Güvenilirlik: {signal_strength}/100
{risk_level}

Kripto Çifti: {symbol}
Fiyat: {price_str}

⏰ Zaman Dilimleri:
1 Saat: {signal_1h}
2 Saat: {signal_2h}
1 Gün: {signal_1d}

Kaldıraç Önerisi: {leverage_suggestion}

💰 Hedef Fiyat: {target_price_str}
🛑 Stop Loss: {stop_loss_str}

⚠️ YATIRIM TAVSİYESİ DEĞİLDİR ⚠️

📋 DİKKAT:
• Portföyünüzün max %5-10'unu kullanın
• Stop loss'u mutlaka uygulayın
• FOMO ile acele karar vermeyin
• Hedef fiyata ulaşınca kar alın
• Kendi araştırmanızı yapın
"""
    return message, dominant_signal, target_price, stop_loss, stop_loss_str

async def async_get_historical_data(symbol, interval, lookback):
    """Binance'den geçmiş verileri asenkron çek"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={lookback}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=False) as resp:
            klines = await resp.json()
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df

def calculate_full_pine_signals(df, timeframe):
    """
    Pine Script’teki Sinyal Koşulları ve Değerler
    1. RSI
       Periyot:
         1w = 28
         1d = 21
         4h = 18
         diğer = 14
       Eşikler:
         rsiOversold = 40
         rsiOverbought = 60
    2. MACD
       Fast:
         1w = 18
         1d = 13
         4h = 11
         diğer = 10
       Slow:
         1w = 36
         1d = 26
         4h = 22
         diğer = 20
       Signal:
         1w = 12
         1d = 10
         4h = 8
         diğer = 9
    3. Supertrend
       ATR periyot:
         4h = 7
         diğer = 10
       ATR multiplier:
         1w = atrDynamic / 2
         1d = atrDynamic / 1.2
         4h = atrDynamic / 1.3
         diğer = atrDynamic / 1.5
    4. EMA (MA)
       shortMA:
         1w = 30
         1d = 20
         4h = 12
         diğer = 9
       longMA:
         1w = 150
         1d = 100
         4h = 60
         diğer = 50
    5. MFI
       Periyot:
         1w = 25
         1d = 20
         4h = 16
         diğer = 14
       Bullish: < 65
       Bearish: > 35
    """
    # Zaman dilimine göre parametreler
    is_higher_tf = timeframe in ['1d', '4h', '1w']
    is_weekly = timeframe == '1w'
    is_daily = timeframe == '1d'
    is_4h = timeframe == '4h'
    rsi_length = 28 if is_weekly else 21 if is_daily else 18 if is_4h else 14
    macd_fast = 18 if is_weekly else 13 if is_daily else 11 if is_4h else 10
    macd_slow = 36 if is_weekly else 26 if is_daily else 22 if is_4h else 20
    macd_signal = 12 if is_weekly else 10 if is_daily else 8 if is_4h else 9
    short_ma_period = 30 if is_weekly else 20 if is_daily else 12 if is_4h else 9
    long_ma_period = 150 if is_weekly else 100 if is_daily else 60 if is_4h else 50
    mfi_length = 25 if is_weekly else 20 if is_daily else 16 if is_4h else 14
    fib_lookback = 150 if is_weekly else 100 if is_daily else 70 if is_4h else 50

    # Trend
    df['ema200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    df['trend_bullish'] = df['close'] > df['ema200']
    df['trend_bearish'] = df['close'] < df['ema200']

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_length).rsi()
    rsi_overbought = 60
    rsi_oversold = 40

    # MACD
    macd = ta.trend.MACD(df['close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Supertrend (Pine Script ile aynı mantıkta, direction ve value döndürmeli)
    def supertrend(df, atr_period, atr_multiplier):
        hl2 = (df['high'] + df['low']) / 2
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period).average_true_range()
        atr_dynamic = atr.rolling(window=5).mean()
        upperband = hl2 + (atr_multiplier * atr_dynamic)
        lowerband = hl2 - (atr_multiplier * atr_dynamic)
        direction = [1]
        for i in range(1, len(df)):
            prev_dir = direction[i-1]
            if prev_dir == 1:
                if df['close'].iloc[i] <= lowerband.iloc[i-1]:
                    direction.append(-1)
                else:
                    direction.append(1)
            else:
                if df['close'].iloc[i] >= upperband.iloc[i-1]:
                    direction.append(1)
                else:
                    direction.append(-1)
        return pd.Series(direction, index=df.index)
    atr_period = 7 if is_4h else 10
    atr_dynamic = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period).average_true_range().rolling(window=5).mean()
    atr_multiplier = atr_dynamic / 2 if is_weekly else atr_dynamic / 1.2 if is_daily else atr_dynamic / 1.3 if is_4h else atr_dynamic / 1.5
    df['supertrend_dir'] = supertrend(df, atr_period, atr_multiplier.bfill())

    # Hareketli Ortalamalar
    df['short_ma'] = ta.trend.EMAIndicator(df['close'], window=short_ma_period).ema_indicator()
    df['long_ma'] = ta.trend.EMAIndicator(df['close'], window=long_ma_period).ema_indicator()
    df['ma_bullish'] = df['short_ma'] > df['long_ma']
    df['ma_bearish'] = df['short_ma'] < df['long_ma']

    # Hacim Analizi
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['enough_volume'] = df['volume'] > df['volume_ma'] * (0.15 if is_higher_tf else 0.4)

    # MFI (Pine Script ile aynı mantıkta, cumulative valuewhen ile)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    price_up = typical_price > typical_price.shift(1)
    price_down = typical_price < typical_price.shift(1)
    positive_flow = (money_flow * price_up).replace(False, 0).cumsum()
    negative_flow = (money_flow * price_down).replace(False, 0).cumsum()
    df['mfi'] = 100 - 100 / (1 + positive_flow / negative_flow)
    df['mfi_bullish'] = df['mfi'] < 65
    df['mfi_bearish'] = df['mfi'] > 35

    # Fibonacci Filtresi
    fib_level1 = df['high'].rolling(window=fib_lookback).max() * 0.618
    fib_level2 = df['low'].rolling(window=fib_lookback).min() * 1.382
    df['fib_in_range'] = True  # fibFilterEnabled kapalıysa hep True

    # --- PineScript ile birebir AL/SAT/NÖTR sinyal mantığı ---
    def crossover(series1, series2):
        return (series1.shift(1) < series2.shift(1)) & (series1 > series2)
    def crossunder(series1, series2):
        return (series1.shift(1) > series2.shift(1)) & (series1 < series2)

    buy_signal = (
        crossover(df['macd'], df['macd_signal']) |
        (
            (df['rsi'] < rsi_oversold) &
            (df['supertrend_dir'] == 1) &
            (df['ma_bullish']) &
            (df['enough_volume']) &
            (df['mfi_bullish']) &
            (df['trend_bullish'])
        )
    ) & df['fib_in_range']

    sell_signal = (
        crossunder(df['macd'], df['macd_signal']) |
        (
            (df['rsi'] > rsi_overbought) &
            (df['supertrend_dir'] == -1) &
            (df['ma_bearish']) &
            (df['enough_volume']) &
            (df['mfi_bearish']) &
            (df['trend_bearish'])
        )
    ) & df['fib_in_range']

    df['signal'] = 0
    df.loc[buy_signal, 'signal'] = 1
    df.loc[sell_signal, 'signal'] = -1
    # Son bar için nötr (0) sinyal ataması kaldırıldı, sadece AL veya SAT olacak
    return df

# --- YENİ ANA DÖNGÜ VE MANTIK ---
async def get_active_high_volume_usdt_pairs(min_volume=MIN_VOLUME):
    """
    Sadece spotta aktif, USDT bazlı ve 24s hacmi min_volume üstü tüm coinleri döndürür.
    1 günlük (1d) verisi 30'dan az olan yeni coinler otomatik olarak atlanır.
    USDCUSDT, FDUSDUSDT gibi 1:1 stablecoin çiftleri hariç tutulur.
    Ek güvenlik: Son 7 günde %40+ düşüş yaşayan coinler hariç tutulur.
    """
    exchange_info = client.get_exchange_info()
    tickers = client.get_ticker()
    spot_usdt_pairs = set()
    for symbol in exchange_info['symbols']:
        if (
            symbol['quoteAsset'] == 'USDT' and
            symbol['status'] == 'TRADING' and
            symbol['isSpotTradingAllowed']
        ):
            spot_usdt_pairs.add(symbol['symbol'])
    # Hacim kontrolü ve sıralama
    high_volume_pairs = []
    for ticker in tickers:
        symbol = ticker['symbol']
        # Stablecoin çiftlerini hariç tut
        if symbol in STABLECOINS:
            continue
        if symbol in spot_usdt_pairs:
            try:
                quote_volume = float(ticker['quoteVolume'])
                if quote_volume >= min_volume:
                    high_volume_pairs.append((symbol, quote_volume))
            except Exception:
                continue
    # Hacme göre sırala
    high_volume_pairs.sort(key=lambda x: x[1], reverse=True)
    # 1d verisi 30'dan az olanları atla, uygun tüm coinleri döndür
    uygun_pairs = []
    for symbol, volume in high_volume_pairs:
        try:
            df_1d = await async_get_historical_data(symbol, '1d', 40)
            if len(df_1d) < MIN_DAYS:
                print(f"{symbol}: 1d veri yetersiz ({len(df_1d)})")
                continue  # yeni coin, atla
            
            # Son 7 günde %40+ düşüş kontrolü
            if len(df_1d) >= 7:
                current_price = float(df_1d['close'].iloc[-1])
                week_ago_price = float(df_1d['close'].iloc[-8])  # 7 gün önce
                price_change_percent = ((current_price - week_ago_price) / week_ago_price) * 100
                
                if price_change_percent < -WEEKLY_LIMIT:
                    continue
            
            uygun_pairs.append(symbol)
        except Exception as e:
            print(f"{symbol}: 1d veri çekilemedi: {e}")
            continue
    return uygun_pairs

async def main():
    sent_signals = dict()  # {(symbol, sinyal_tipi): signal_values}
    positions = dict()  # {symbol: position_info}
    cooldown_signals = dict()  # {(symbol, sinyal_tipi): datetime}
    stop_cooldown = dict()  # {symbol: datetime}
    previous_signals = dict()  # {symbol: {tf: signal}} - İlk çalıştığında kaydedilen sinyaller
    stopped_coins = dict()  # {symbol: {...}}
    active_signals = dict()  # {symbol: {...}} - Aktif sinyaller
    successful_signals = dict()  # {symbol: {...}} - Başarılı sinyaller (hedefe ulaşan)
    failed_signals = dict()  # {symbol: {...}} - Başarısız sinyaller (stop olan)
    tracked_coins = set()  # Takip edilen tüm coinlerin listesi
    first_run = True  # İlk çalıştırma kontrolü
    pending_signals = dict()  # {(symbol, sinyal_tipi): {"signal_values": [...], "timestamp": datetime, "signals": {...}}}
    
    # Genel istatistikler
    stats = {
        "total_signals": 0,
        "successful_signals": 0,
        "failed_signals": 0,
        "total_profit_loss": 0.0,  # 100$ yatırım için
        "active_signals_count": 0,
        "tracked_coins_count": 0
    }
    
    timeframes = {
        '1h': '1h',
        '2h': '2h',
        '1d': '1d'
    }
    tf_names = ['1h', '2h', '1d']
    
    print("Sinyal botu başlatıldı!")
    print("İlk çalıştırma: Mevcut sinyaller kaydediliyor, değişiklik bekleniyor...")
    
    while True:
        try:
            symbols = await get_active_high_volume_usdt_pairs(min_volume=MIN_VOLUME)
            tracked_coins.update(symbols)  # Takip edilen coinleri güncelle
            print(f"Takip edilen coin sayısı: {len(symbols)}")
            
            # 1. Pozisyonları kontrol et (hedef/stop)
            for symbol, pos in list(positions.items()):
                try:
                    df = await async_get_historical_data(symbol, '1h', 2)  # En güncel fiyatı çek
                    last_price = float(df['close'].iloc[-1])
                    
                    # Aktif sinyal bilgilerini güncelle
                    if symbol in active_signals:
                        active_signals[symbol]["current_price"] = format_price(last_price, pos["open_price"])
                        active_signals[symbol]["current_price_float"] = last_price
                        active_signals[symbol]["last_update"] = str(datetime.now())
                    
                    if pos["type"] == "ALIŞ":
                        if last_price >= pos["target"]:
                            msg = f"🎯 <b>HEDEF BAŞARIYLA GERÇEKLEŞTİ!</b> 🎯\n\n<b>{symbol}</b> işlemi için hedef fiyatına ulaşıldı!\nÇıkış Fiyatı: <b>{format_price(last_price)}</b>\n"
                            await send_telegram_message(msg)
                            cooldown_signals[(symbol, "ALIS")] = datetime.now()
                            
                            # Başarılı sinyal olarak kaydet
                            profit_percent = 2
                            profit_usd = 100 * 0.02 * 10
                            successful_signals[symbol] = {
                                "symbol": symbol,
                                "type": pos["type"],
                                "entry_price": format_price(pos["open_price"], pos["open_price"]),
                                "exit_price": format_price(last_price, pos["open_price"]),
                                "target_price": format_price(pos["target"], pos["open_price"]),
                                "stop_loss": format_price(pos["stop"], pos["open_price"]),
                                "signals": pos["signals"],
                                "completion_time": str(datetime.now()),
                                "status": "SUCCESS",
                                "profit_percent": round(profit_percent, 2),
                                "profit_usd": round(profit_usd, 2),
                                "leverage": pos.get("leverage", 1),
                                "entry_time": pos.get("entry_time", str(datetime.now())),
                                "duration_hours": round((datetime.now() - datetime.fromisoformat(pos.get("entry_time", str(datetime.now())))).total_seconds() / 3600, 2)
                            }
                            
                            # İstatistikleri güncelle
                            stats["successful_signals"] += 1
                            stats["total_profit_loss"] += profit_usd
                            
                            if symbol in active_signals:
                                del active_signals[symbol]
                            
                            del positions[symbol]
                        elif last_price <= pos["stop"]:
                            msg = f"❌ {symbol} işlemi stop oldu! Stop fiyatı: {pos['stop_str']}, Şu anki fiyat: {format_price(last_price, pos['stop'])}"
                            await send_telegram_message(msg)
                            cooldown_signals[(symbol, "ALIS")] = datetime.now()
                            stop_cooldown[symbol] = datetime.now()
                            
                            # Stop olan coini stopped_coins'e ekle (tüm detaylarla)
                            stopped_coins[symbol] = {
                                "symbol": symbol,
                                "type": pos["type"],
                                "entry_price": format_price(pos["open_price"], pos["open_price"]),
                                "stop_time": str(datetime.now()),
                                "target_price": format_price(pos["target"], pos["open_price"]),
                                "stop_loss": format_price(pos["stop"], pos["open_price"]),
                                "signals": pos["signals"],
                                "min_price": format_price(last_price, pos["open_price"]),
                                "max_drawdown_percent": 0.0,
                                "reached_target": False
                            }
                            
                            # Başarısız sinyal olarak kaydet
                            loss_percent = -1
                            loss_usd = -100 * 0.01 * 10
                            failed_signals[symbol] = {
                                "symbol": symbol,
                                "type": pos["type"],
                                "entry_price": format_price(pos["open_price"], pos["open_price"]),
                                "exit_price": format_price(last_price, pos["open_price"]),
                                "target_price": format_price(pos["target"], pos["open_price"]),
                                "stop_loss": format_price(pos["stop"], pos["open_price"]),
                                "signals": pos["signals"],
                                "completion_time": str(datetime.now()),
                                "status": "FAILED",
                                "loss_percent": round(loss_percent, 2),
                                "loss_usd": round(loss_usd, 2),
                                "leverage": pos.get("leverage", 1),
                                "entry_time": pos.get("entry_time", str(datetime.now())),
                                "duration_hours": round((datetime.now() - datetime.fromisoformat(pos.get("entry_time", str(datetime.now())))).total_seconds() / 3600, 2)
                            }
                            
                            # İstatistikleri güncelle
                            stats["failed_signals"] += 1
                            stats["total_profit_loss"] += loss_usd
                            
                            if symbol in active_signals:
                                del active_signals[symbol]
                            
                            del positions[symbol]
                    elif pos["type"] == "SATIŞ":
                        if last_price <= pos["target"]:
                            msg = f"🎯 <b>HEDEF BAŞARIYLA GERÇEKLEŞTİ!</b> 🎯\n\n<b>{symbol}</b> işlemi için hedef fiyatına ulaşıldı!\nÇıkış Fiyatı: <b>{format_price(last_price)}</b>\n"
                            await send_telegram_message(msg)
                            cooldown_signals[(symbol, "SATIS")] = datetime.now()
                            
                            # Başarılı sinyal olarak kaydet
                            profit_percent = 2
                            profit_usd = 100 * 0.02 * 10
                            successful_signals[symbol] = {
                                "symbol": symbol,
                                "type": pos["type"],
                                "entry_price": format_price(pos["open_price"], pos["open_price"]),
                                "exit_price": format_price(last_price, pos["open_price"]),
                                "target_price": format_price(pos["target"], pos["open_price"]),
                                "stop_loss": format_price(pos["stop"], pos["open_price"]),
                                "signals": pos["signals"],
                                "completion_time": str(datetime.now()),
                                "status": "SUCCESS",
                                "profit_percent": round(profit_percent, 2),
                                "profit_usd": round(profit_usd, 2),
                                "leverage": pos.get("leverage", 1),
                                "entry_time": pos.get("entry_time", str(datetime.now())),
                                "duration_hours": round((datetime.now() - datetime.fromisoformat(pos.get("entry_time", str(datetime.now())))).total_seconds() / 3600, 2)
                            }
                            
                            # İstatistikleri güncelle
                            stats["successful_signals"] += 1
                            stats["total_profit_loss"] += profit_usd
                            
                            if symbol in active_signals:
                                del active_signals[symbol]
                            
                            del positions[symbol]
                except Exception as e:
                    print(f"Pozisyon kontrol hatası: {symbol} - {str(e)}")
                    continue
            
            # 2. Sinyal arama
            async def process_symbol(symbol):
                # Eğer pozisyon açıksa, yeni sinyal arama
                if symbol in positions:
                    return
                # Stop sonrası 2 saatlik cooldown kontrolü
                if symbol in stop_cooldown:
                    last_stop = stop_cooldown[symbol]
                    if (datetime.now() - last_stop) < timedelta(hours=COOLDOWN):
                        return  # 2 saat dolmadıysa sinyal arama
                    else:
                        del stop_cooldown[symbol]  # 2 saat dolduysa tekrar sinyal aranabilir
                        print(f"{symbol} için stop sonrası cooldown bitti, tekrar sinyal aranacak.")
                # 1 günlük veri kontrolü
                try:
                    df_1d = await async_get_historical_data(symbol, timeframes['1d'], 40)
                    if len(df_1d) < 30:
                        print(f"UYARI: {symbol} için 1 günlük veri 30'dan az, sinyal aranmıyor.")
                        return
                except Exception as e:
                    print(f"UYARI: {symbol} için 1 günlük veri çekilemedi: {str(e)}")
                    return
                # Mevcut sinyalleri al
                current_signals = dict()
                for tf_name in tf_names:
                    try:
                        df = await async_get_historical_data(symbol, timeframes[tf_name], 200)
                        df = calculate_full_pine_signals(df, tf_name)
                        current_signals[tf_name] = int(df['signal'].iloc[-1])
                    except Exception as e:
                        print(f"Hata: {symbol} - {tf_name} - {str(e)}")
                        current_signals[tf_name] = 0
                # İlk çalıştırmada sadece sinyalleri kaydet
                if first_run:
                    previous_signals[symbol] = current_signals.copy()
                    print(f"İlk çalıştırma - {symbol} sinyalleri kaydedildi: {current_signals}")
                    return
                # İlk çalıştırma değilse, değişiklik kontrolü yap
                if symbol in previous_signals:
                    prev_signals = previous_signals[symbol]
                    signal_changed = False
                    # Herhangi bir zaman diliminde değişiklik var mı kontrol et
                    for tf in tf_names:
                        if prev_signals[tf] != current_signals[tf]:
                            # Sadece -1 <-> 1 değişimlerinde logla, 0 (nötr) ise atla
                            if (prev_signals[tf] in [-1, 1]) and (current_signals[tf] in [-1, 1]):
                                signal_changed = True
                                print(f"{symbol} - {tf} sinyali değişti: {prev_signals[tf]} -> {current_signals[tf]}")
                            break
                    if not signal_changed:
                        return  # Değişiklik yoksa veya nötr ise devam et
                    # Değişiklik varsa, yeni sinyal analizi yap
                    signal_values = [current_signals[tf] for tf in tf_names]
                    # Sinyal koşulu: 3 zaman dilimi de aynı olmalı VE güçlü sinyal olmalı
                    if all(s == 1 for s in signal_values):
                        sinyal_tipi = 'ALIS'
                    elif all(s == -1 for s in signal_values):
                        sinyal_tipi = 'SATIS'
                    else:
                        previous_signals[symbol] = current_signals.copy()
                        # Eğer pending_signals'da varsa, iptal et
                        for st in ['ALIS', 'SATIS']:
                            pending_key = (symbol, st)
                            if pending_key in pending_signals:
                                del pending_signals[pending_key]
                        return
                    # Ek güvenlik: Son 24 saatte %8+ hareket varsa sinyal verme
                    try:
                        df_24h = await async_get_historical_data(symbol, '1h', 24)
                        if len(df_24h) >= 24:
                            current_price = float(df_24h['close'].iloc[-1])
                            day_ago_price = float(df_24h['close'].iloc[0])
                            daily_change = abs((current_price - day_ago_price) / day_ago_price) * 100
                            if daily_change > DAILY_LIMIT:
                                print(f"{symbol}: Son 24 saatte %{daily_change:.1f} hareket, sinyal atlanıyor")
                                previous_signals[symbol] = current_signals.copy()
                                # pending_signals'dan çıkar
                                for st in ['ALIS', 'SATIS']:
                                    pending_key = (symbol, st)
                                    if pending_key in pending_signals:
                                        del pending_signals[pending_key]
                                return
                    except Exception as e:
                        print(f"{symbol}: 24h veri kontrolü hatası: {e}")
                        return
                    # 4 saatlik cooldown kontrolü
                    cooldown_key = (symbol, sinyal_tipi)
                    if cooldown_signals:
                        last_time = cooldown_signals.get(cooldown_key)
                        if last_time and (datetime.now() - last_time) < timedelta(hours=COOLDOWN):
                            previous_signals[symbol] = current_signals.copy()
                            return
                        elif last_time:
                            del cooldown_signals[cooldown_key]
                    # Aynı sinyal daha önce gönderilmiş mi kontrol et
                    signal_key = (symbol, sinyal_tipi)
                    if sent_signals.get(signal_key) == signal_values:
                        previous_signals[symbol] = current_signals.copy()
                        return
                    # --- YENİ: Beklemeye al ---
                    if signal_key not in pending_signals:
                        print(f"{symbol} için sinyal beklemeye alındı: {sinyal_tipi} - {current_signals}")
                        pending_signals[signal_key] = {
                            "signal_values": signal_values.copy(),
                            "timestamp": datetime.now(),
                            "signals": current_signals.copy()
                        }
                    else:
                        # Zaten beklemede, güncelle
                        pending_signals[signal_key]["signal_values"] = signal_values.copy()
                        pending_signals[signal_key]["signals"] = current_signals.copy()
                    previous_signals[symbol] = current_signals.copy()
                    return  # HEMEN SİNYAL GÖNDERME!
                await asyncio.sleep(0)  # Task'ler arası context switch için

            # Paralel task listesi oluştur
            tasks = [process_symbol(symbol) for symbol in symbols]
            await asyncio.gather(*tasks)
            
            # İlk çalıştırma tamamlandıysa
            if first_run:
                first_run = False
                print("İlk çalıştırma tamamlandı! Artık değişiklikler takip ediliyor...")
            
            # Aktif sinyallerin fiyatlarını güncelle
            for symbol in list(active_signals.keys()):
                if symbol not in positions:  # Pozisyon kapandıysa aktif sinyalden kaldır
                    del active_signals[symbol]
                    continue
                try:
                    df = await async_get_historical_data(symbol, '1h', 2)
                    last_price = float(df['close'].iloc[-1])
                    active_signals[symbol]["current_price"] = format_price(last_price, active_signals[symbol]["entry_price_float"])
                    active_signals[symbol]["current_price_float"] = last_price
                    active_signals[symbol]["last_update"] = str(datetime.now())
                except Exception as e:
                    print(f"Aktif sinyal güncelleme hatası: {symbol} - {str(e)}")
                    continue
            
            # İstatistikleri güncelle
            stats["active_signals_count"] = len(active_signals)
            stats["tracked_coins_count"] = len(tracked_coins)
            
            # Kaydetme fonksiyonları kaldırıldı
            
            # STOP OLAN COINLERİ TAKİP ET
            for symbol, info in list(stopped_coins.items()):
                try:
                    df = await async_get_historical_data(symbol, '1h', 2)
                    last_price = float(df['close'].iloc[-1])
                    entry_price = float(info["entry_price"])
                    if info["type"] == "ALIŞ":
                        # Min fiyatı güncelle
                        min_price = float(info["min_price"])
                        if last_price < min_price:
                            min_price = last_price
                        info["min_price"] = format_price(min_price, entry_price)
                        # Max terse gidiş (drawdown)
                        drawdown = (min_price - entry_price) / entry_price * 100
                        if drawdown < float(info.get("max_drawdown_percent", 0.0)):
                            info["max_drawdown_percent"] = round(drawdown, 2)
                        else:
                            info["max_drawdown_percent"] = round(float(info.get("max_drawdown_percent", drawdown)), 2)
                        # Hedefe ulaşıldı mı?
                        if not info["reached_target"] and last_price >= float(info["target_price"]):
                            info["reached_target"] = True
                        # Kaydetme fonksiyonu kaldırıldı
                        if info["reached_target"]:
                            del stopped_coins[symbol]
                    elif info["type"] == "SATIŞ":
                        # Max fiyatı güncelle
                        max_price = float(info["max_price"])
                        if last_price > max_price:
                            max_price = last_price
                        info["max_price"] = format_price(max_price, entry_price)
                        # Max terse gidiş (drawup)
                        drawup = (max_price - entry_price) / entry_price * 100
                        if drawup > float(info.get("max_drawup_percent", 0.0)):
                            info["max_drawup_percent"] = round(drawup, 2)
                        else:
                            info["max_drawup_percent"] = round(float(info.get("max_drawup_percent", drawup)), 2)
                        # Hedefe ulaşıldı mı?
                        if not info["reached_target"] and last_price <= float(info["target_price"]):
                            info["reached_target"] = True
                        # Kaydetme fonksiyonu kaldırıldı
                        if info["reached_target"]:
                            del stopped_coins[symbol]
                except Exception as e:
                    print(f"Stop sonrası takip hatası: {symbol} - {str(e)}")
                    continue
            
            # --- PENDING SİNYALLERİ KONTROL ET ---
            for pending_key, pending_info in list(pending_signals.items()):
                symbol, sinyal_tipi = pending_key
                bekleme_suresi = (datetime.now() - pending_info["timestamp"]).total_seconds()
                if bekleme_suresi < 1800:  # 30 dakika = 1800 saniye
                    continue  # Bekleme süresi dolmadı
                # 30 dakika doldu, sinyal hala aynı mı kontrol et
                try:
                    current_signals = dict()
                    for tf_name in tf_names:
                        df = await async_get_historical_data(symbol, timeframes[tf_name], 200)
                        df = calculate_full_pine_signals(df, tf_name)
                        current_signals[tf_name] = int(df['signal'].iloc[-1])
                    signal_values = [current_signals[tf] for tf in tf_names]
                    if signal_values == pending_info["signal_values"]:
                        # Sinyal hala aynı, artık sinyal gönder
                        price = float(df['close'].iloc[-1])
                        signal_strength = calculate_signal_strength(df, current_signals)
                        if signal_strength < MIN_STRENGTH:
                            print(f"{symbol}: Sinyal gücü düşük ({signal_strength}/100), bekleyen sinyal iptal edildi")
                            del pending_signals[pending_key]
                            continue
                        message, dominant_signal, target_price, stop_loss, stop_loss_str = create_signal_message(symbol, price, current_signals, signal_strength)
                        if message:
                            print(f"30dk sonra Telegram'a gönderiliyor: {symbol} - {dominant_signal} (Güç: {signal_strength}/100)")
                            await send_telegram_message(message)
                            # Kaldıraç hesaplama (güce göre)
                            if signal_strength >= HIGH_STRENGTH:
                                leverage = HIGH_LEV
                            elif signal_strength >= MIN_STRENGTH:
                                leverage = MED_LEV
                            else:
                                leverage = LOW_LEV
                            positions[symbol] = {
                                "type": dominant_signal,
                                "target": float(target_price),
                                "stop": float(stop_loss),
                                "open_price": float(price),
                                "stop_str": stop_loss_str,
                                "signals": {k: ("ALIŞ" if v == 1 else "SATIŞ") for k, v in current_signals.items()},
                                "leverage": leverage,
                                "signal_strength": signal_strength,
                                "entry_time": str(datetime.now())
                            }
                            active_signals[symbol] = {
                                "symbol": symbol,
                                "type": dominant_signal,
                                "entry_price": format_price(price, price),
                                "entry_price_float": price,
                                "target_price": format_price(target_price, price),
                                "stop_loss": format_price(stop_loss, price),
                                "signals": {k: ("ALIŞ" if v == 1 else "SATIŞ") for k, v in current_signals.items()},
                                "leverage": leverage,
                                "signal_strength": signal_strength,
                                "signal_time": str(datetime.now()),
                                "current_price": format_price(price, price),
                                "current_price_float": price,
                                "last_update": str(datetime.now())
                            }
                            stats["total_signals"] += 1
                            stats["active_signals_count"] = len(active_signals)
                        del pending_signals[pending_key]
                    else:
                        print(f"{symbol}: Bekleyen sinyal değişti, iptal edildi.")
                        del pending_signals[pending_key]
                except Exception as e:
                    print(f"Bekleyen sinyal kontrol hatası: {symbol} - {str(e)}")
                    del pending_signals[pending_key]
            
            # İstatistik özeti yazdır
            print(f"📊 İSTATİSTİK ÖZETİ:")
            print(f"   Toplam Sinyal: {stats['total_signals']}")
            print(f"   Başarılı: {stats['successful_signals']}")
            print(f"   Başarısız: {stats['failed_signals']}")
            print(f"   Aktif Sinyal: {stats['active_signals_count']}")
            print(f"   Toplam Görülen Coin: {stats['tracked_coins_count']}")
            print(f"   100$ Yatırım Toplam Kar/Zarar: ${stats['total_profit_loss']:.2f}")
            # Sadece kapanmış işlemler için ortalama kar/zarar
            closed_count = stats['successful_signals'] + stats['failed_signals']
            closed_pl = 0.0
            for s in successful_signals.values():
                closed_pl += s.get('profit_usd', 0)
            for f in failed_signals.values():
                closed_pl += f.get('loss_usd', 0)
            if closed_count > 0:
                avg_closed_pl = closed_pl / closed_count
                success_rate = (stats['successful_signals'] / closed_count) * 100
                print(f"   Başarı Oranı: %{success_rate:.1f}")
            else:
                print(f"   Başarı Oranı: %0.0")
            # Döngü sonunda bekleme süresi
            print("Tüm coinler kontrol edildi. 30 saniye bekleniyor...")
            await asyncio.sleep(30)
            
            # Kaydetme fonksiyonu kaldırıldı
            
        except Exception as e:
            print(f"Genel hata: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
