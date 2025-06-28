import asyncio
from collections import defaultdict
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime, timedelta
import telegram
import requests
import certifi
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# SSL uyarılarını kapat
urllib3.disable_warnings(InsecureRequestWarning)

# Telegram Bot ayarları
TELEGRAM_TOKEN = "7872345042:AAE6Om2LGtz1QjqfZz8ge0em6Gw29llzFno"
TELEGRAM_CHAT_ID = "847081095"

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
            # price'ı stringe çevirip, noktadan sonra dec kadar basamak al
            price_str = f"{price:.{dec+8}f}"  # fazladan hassasiyet, sonra kısalt
            int_part, frac_part = price_str.split('.')
            frac_part = frac_part[:dec]
            # Eğer dec>0 ise, ondalık kısmı sıfır da olsa göster
            return f"{int_part}.{frac_part}" if dec > 0 else int_part
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

def create_signal_message(symbol, price, signals):
    """Sinyal mesajını oluştur (AL/SAT başlıkta)"""
    price_str = format_price(price, price)  # Fiyatın kendi basamağı kadar
    signal_30m = "ALIŞ" if signals['30m'] == 1 else "SATIŞ" if signals['30m'] == -1 else "NÖTR"
    signal_2h = "ALIŞ" if signals['2h'] == 1 else "SATIŞ" if signals['2h'] == -1 else "NÖTR"
    signal_1d = "ALIŞ" if signals['1d'] == 1 else "SATIŞ" if signals['1d'] == -1 else "NÖTR"
    buy_count = sum(1 for s in signals.values() if s == 1)
    sell_count = sum(1 for s in signals.values() if s == -1)
    if buy_count >= 2:
        dominant_signal = "ALIŞ"
        target_price = price * 1.01  # %1 hedef
        stop_loss = price * 0.995    # %0.5 stop
        sinyal_tipi = "AL SİNYALİ"
    elif sell_count >= 2:
        dominant_signal = "SATIŞ"
        target_price = price * 0.99  # %1 hedef
        stop_loss = price * 1.005    # %0.5 stop
        sinyal_tipi = "SAT SİNYALİ"
    else:
        return None, None, None, None  # Çoğunluk sinyali yoksa mesaj gönderme
    leverage = 15 if buy_count == 3 or sell_count == 3 else 10
    target_price_str = format_price(target_price, price)
    stop_loss_str = format_price(stop_loss, price)
    message = f"""
🚨 {sinyal_tipi} 

Kripto Çifti: {symbol}
Fiyat: {price_str}

⏰ Zaman Dilimleri:
30 Dakika: {signal_30m}
2 Saat: {signal_2h}
1 Gün: {signal_1d}

📊 Kaldıraç Önerisi: {leverage}x

💰 Hedef Fiyat: {target_price_str}
🛑 Stop Loss: {stop_loss_str}

⚠️ YATIRIM TAVSİYESİ DEĞİLDİR ⚠️
"""
    return message, dominant_signal, target_price, stop_loss, stop_loss_str

def get_historical_data(symbol, interval, lookback):
    """Binance'den geçmiş verileri çek"""
    klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=lookback
    )
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

def calculate_full_pine_signals(df, timeframe, fib_filter_enabled=False):
    """
    Pine Script algo.pine mantığını eksiksiz şekilde Python'a taşır.
    df: pandas DataFrame (timestamp, open, high, low, close, volume)
    timeframe: '15m', '2h', '1d', '1w' gibi string
    fib_filter_enabled: Fibonacci filtresi aktif mi?
    Dönüş: df (ekstra sütunlarla, en sonda 'signal')
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

    # EMA ve trend
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

    # Supertrend (özel fonksiyon)
    def supertrend(df, atr_period, multiplier):
        hl2 = (df['high'] + df['low']) / 2
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period).average_true_range()
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        direction = [1]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upperband.iloc[i-1]:
                direction.append(1)
            elif df['close'].iloc[i] < lowerband.iloc[i-1]:
                direction.append(-1)
            else:
                direction.append(direction[-1])
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
    volume_ma_period = 20
    df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
    df['enough_volume'] = df['volume'] > df['volume_ma'] * (0.15 if is_higher_tf else 0.4)

    # MFI
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=mfi_length).money_flow_index()
    df['mfi_bullish'] = df['mfi'] < 65
    df['mfi_bearish'] = df['mfi'] > 35

    # Fibonacci Seviyeleri
    highest_high = df['high'].rolling(window=fib_lookback).max()
    lowest_low = df['low'].rolling(window=fib_lookback).min()
    fib_level1 = highest_high * 0.618
    fib_level2 = lowest_low * 1.382
    if fib_filter_enabled:
        df['fib_in_range'] = (df['close'] > fib_level1) & (df['close'] < fib_level2)
    else:
        df['fib_in_range'] = True

    # Ichimoku Bulutu (isteğe bağlı, sinyalde kullanılmıyor ama eklenebilir)
    def ichimoku(df, conv_periods=9, base_periods=26, span_b_periods=52, displacement=26):
        high = df['high']
        low = df['low']
        close = df['close']
        conv_line = (high.rolling(window=conv_periods).max() + low.rolling(window=conv_periods).min()) / 2
        base_line = (high.rolling(window=base_periods).max() + low.rolling(window=base_periods).min()) / 2
        leading_span_a = ((conv_line + base_line) / 2).shift(displacement)
        leading_span_b = ((high.rolling(window=span_b_periods).max() + low.rolling(window=span_b_periods).min()) / 2).shift(displacement)
        lagging_span = close.shift(-displacement)
        return conv_line, base_line, leading_span_a, leading_span_b, lagging_span
    # conv_line, base_line, leading_span_a, leading_span_b, lagging_span = ichimoku(df)

    # Pivot Noktaları (isteğe bağlı, sinyalde kullanılmıyor ama eklenebilir)
    def pivot_points(df):
        high = df['high'].shift(1)
        low = df['low'].shift(1)
        close = df['close'].shift(1)
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        return pivot, r1, s1, r2, s2
    # df['pivot'], df['r1'], df['s1'], df['r2'], df['s2'] = pivot_points(df)

    # AL/SAT Koşulları
    buy_signal = (
        (df['macd'] > df['macd_signal']) |
        ((df['rsi'] < rsi_oversold) &
         (df['supertrend_dir'] == 1) &
         (df['ma_bullish']) &
         (df['enough_volume']) &
         (df['mfi_bullish']) &
         (df['trend_bullish']))
    ) & df['fib_in_range']

    sell_signal = (
        (df['macd'] < df['macd_signal']) |
        ((df['rsi'] > rsi_overbought) &
         (df['supertrend_dir'] == -1) &
         (df['ma_bearish']) &
         (df['enough_volume']) &
         (df['mfi_bearish']) &
         (df['trend_bearish']))
    ) & df['fib_in_range']

    df['signal'] = 0
    df.loc[buy_signal, 'signal'] = 1
    df.loc[sell_signal, 'signal'] = -1

    return df

# --- YENİ ANA DÖNGÜ VE MANTIK ---
async def get_active_high_volume_usdt_pairs(min_volume=1000000, top_n=5):
    """
    Sadece spotta aktif, USDT bazlı ve 24s hacmi min_volume üstü coinlerden en yüksek hacimli top_n tanesini döndürür.
    1 günlük (1d) verisi 30'dan az olan yeni coinler otomatik olarak atlanır.
    USDCUSDT, FDUSDUSDT gibi 1:1 stablecoin çiftleri hariç tutulur.
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
        if symbol in ['USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'BUSDUSDT', 'USDPUSDT', 'USDTUSDT']:
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
    # 1d verisi 30'dan az olanları atla, uygun ilk top_n coini döndür
    uygun_pairs = []
    for symbol, volume in high_volume_pairs:
        try:
            df_1d = get_historical_data(symbol, '1d', 40)
            if len(df_1d) < 30:
                continue  # yeni coin, atla
            uygun_pairs.append(symbol)
            if len(uygun_pairs) == top_n:
                break
        except Exception:
            continue
    return uygun_pairs

async def main():
    sent_signals = dict()  # {(symbol, sinyal_tipi): signal_values}
    positions = dict()  # {symbol: position_info}
    cooldown_signals = dict()  # {(symbol, sinyal_tipi): datetime}
    stop_cooldown = dict()  # {symbol: datetime}
    previous_signals = dict()  # {symbol: {tf: signal}} - İlk çalıştığında kaydedilen sinyaller
    first_run = True  # İlk çalıştırma kontrolü
    
    timeframes = {
        '30m': '30m',
        '2h': '2h',
        '1d': '1d'
    }
    tf_names = ['30m', '2h', '1d']
    
    print("Sinyal botu başlatıldı!")
    print("İlk çalıştırma: Mevcut sinyaller kaydediliyor, değişiklik bekleniyor...")
    
    while True:
        try:
            symbols = await get_active_high_volume_usdt_pairs(top_n=5)
            print(f"Takip edilen en yüksek hacimli 5 USDT çifti: {symbols}")
            
            # 1. Pozisyonları kontrol et (hedef/stop)
            for symbol, pos in list(positions.items()):
                try:
                    df = get_historical_data(symbol, '30m', 2)  # En güncel fiyatı çek
                    last_price = float(df['close'].iloc[-1])
                    if pos["type"] == "ALIŞ":
                        if last_price >= pos["target"]:
                            msg = f"✅ {symbol} işleminden çıkıldı (hedefe ulaşıldı): {format_price(last_price)}"
                            await send_telegram_message(msg)
                            cooldown_signals[(symbol, "ALIS")] = datetime.now()
                            del positions[symbol]
                        elif last_price <= pos["stop"]:
                            current_price_str = format_price(last_price, pos['stop'])
                            msg = f"❌ {symbol} işlemi stop oldu! Stop fiyatı: {pos['stop_str']}, Şu anki fiyat: {current_price_str}"
                            await send_telegram_message(msg)
                            cooldown_signals[(symbol, "ALIS")] = datetime.now()
                            stop_cooldown[symbol] = datetime.now()
                            del positions[symbol]
                    elif pos["type"] == "SATIŞ":
                        if last_price <= pos["target"]:
                            msg = f"✅ {symbol} işleminden çıkıldı (hedefe ulaşıldı): {format_price(last_price)}"
                            await send_telegram_message(msg)
                            cooldown_signals[(symbol, "SATIS")] = datetime.now()
                            del positions[symbol]
                        elif last_price >= pos["stop"]:
                            current_price_str = format_price(last_price, pos['stop'])
                            msg = f"❌ {symbol} işlemi stop oldu! Stop fiyatı: {pos['stop_str']}, Şu anki fiyat: {current_price_str}"
                            await send_telegram_message(msg)
                            cooldown_signals[(symbol, "SATIS")] = datetime.now()
                            stop_cooldown[symbol] = datetime.now()
                            del positions[symbol]
                except Exception as e:
                    print(f"Pozisyon kontrol hatası: {symbol} - {str(e)}")
                    continue
            
            # 2. Sinyal arama
            for symbol in symbols:
                # Eğer pozisyon açıksa, yeni sinyal arama
                if symbol in positions:
                    continue
                
                # Stop sonrası 4 saatlik cooldown kontrolü
                if symbol in stop_cooldown:
                    last_stop = stop_cooldown[symbol]
                    if (datetime.now() - last_stop) < timedelta(hours=4):
                        continue  # 4 saat dolmadıysa sinyal arama
                    else:
                        del stop_cooldown[symbol]  # 4 saat dolduysa tekrar sinyal aranabilir
                
                # 1 günlük veri kontrolü
                try:
                    df_1d = get_historical_data(symbol, timeframes['1d'], 40)
                    if len(df_1d) < 30:
                        print(f"UYARI: {symbol} için 1 günlük veri 30'dan az, sinyal aranmıyor.")
                        continue
                except Exception as e:
                    print(f"UYARI: {symbol} için 1 günlük veri çekilemedi: {str(e)}")
                    continue
                
                # Mevcut sinyalleri al
                current_signals = dict()
                for tf_name in tf_names:
                    try:
                        df = get_historical_data(symbol, timeframes[tf_name], 200)
                        df = calculate_full_pine_signals(df, tf_name)
                        current_signals[tf_name] = int(df['signal'].iloc[-1])
                    except Exception as e:
                        print(f"Hata: {symbol} - {tf_name} - {str(e)}")
                        current_signals[tf_name] = 0
                
                # İlk çalıştırmada sadece sinyalleri kaydet
                if first_run:
                    previous_signals[symbol] = current_signals.copy()
                    print(f"İlk çalıştırma - {symbol} sinyalleri kaydedildi: {current_signals}")
                    continue
                
                # İlk çalıştırma değilse, değişiklik kontrolü yap
                if symbol in previous_signals:
                    prev_signals = previous_signals[symbol]
                    signal_changed = False
                    
                    # Herhangi bir zaman diliminde değişiklik var mı kontrol et
                    for tf in tf_names:
                        if prev_signals[tf] != current_signals[tf]:
                            signal_changed = True
                            print(f"{symbol} - {tf} sinyali değişti: {prev_signals[tf]} -> {current_signals[tf]}")
                            break
                    
                    if not signal_changed:
                        continue  # Değişiklik yoksa devam et
                    
                    # Değişiklik varsa, yeni sinyal analizi yap
                    signal_values = [current_signals[tf] for tf in tf_names]
                    
                    # Sinyal koşullarını kontrol et
                    if all(s == 1 for s in signal_values):
                        sinyal_tipi = 'ALIS'
                    elif all(s == -1 for s in signal_values):
                        sinyal_tipi = 'SATIS'
                    elif (
                        (signal_values[0] == signal_values[1] != 0) or
                        (signal_values[1] == signal_values[2] != 0) or
                        (signal_values[0] == signal_values[2] != 0)
                    ):
                        sinyal_tipi = 'ALIS' if signal_values.count(1) >= 2 else 'SATIS'
                    else:
                        # Sinyal koşulu sağlanmıyorsa sadece güncelle
                        previous_signals[symbol] = current_signals.copy()
                        continue
                    
                    # 4 saatlik cooldown kontrolü
                    cooldown_key = (symbol, sinyal_tipi)
                    if cooldown_key in cooldown_signals:
                        last_time = cooldown_signals[cooldown_key]
                        if (datetime.now() - last_time) < timedelta(hours=4):
                            continue  # 4 saat dolmadıysa sinyal arama
                        else:
                            del cooldown_signals[cooldown_key]  # 4 saat dolduysa tekrar sinyal aranabilir
                    
                    # Aynı sinyal daha önce gönderilmiş mi kontrol et
                    signal_key = (symbol, sinyal_tipi)
                    if sent_signals.get(signal_key) == signal_values:
                        continue
                    
                    # Yeni sinyal gönder
                    sent_signals[signal_key] = signal_values.copy()
                    price = float(df['close'].iloc[-1])
                    message, dominant_signal, target_price, stop_loss, stop_loss_str = create_signal_message(symbol, price, current_signals)
                    
                    if message:
                        print(f"Telegram'a gönderiliyor: {symbol} - {dominant_signal}")
                        print(f"Değişiklik: {prev_signals} -> {current_signals}")
                        await send_telegram_message(message)
                        # Pozisyonu kaydet
                        positions[symbol] = {"type": dominant_signal, "target": target_price, "stop": stop_loss, "open_price": price, "stop_str": stop_loss_str}
                    
                    # Sinyalleri güncelle
                    previous_signals[symbol] = current_signals.copy()
                
                await asyncio.sleep(1)
            
            # İlk çalıştırma tamamlandıysa
            if first_run:
                first_run = False
                print("İlk çalıştırma tamamlandı! Artık değişiklikler takip ediliyor...")
            
            print("Tüm coinler kontrol edildi. 10 saniye bekleniyor...")
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"Genel hata: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
