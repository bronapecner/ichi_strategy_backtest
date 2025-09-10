import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import gspread

# --- GOOGLE CLOUD AUTENTIZACE PRO GOOGLE SHEETS ---
try:
    gc = gspread.service_account(filename='service_account.json')
    logbook_sheet = gc.open("BybitTradeLogbook").sheet1
except Exception as e:
    # Tento print se zobrazí v journalctl, pokud selže start
    print(f"KRITICKÁ CHYBA: Nepodařilo se připojit ke Google Sheets. Zkontrolujte soubor 'service_account.json', název tabulky a její nasdílení. Chyba: {e}")
    sys.exit()

# --- NASTAVENÍ JEDNODUCHÉHO LOGOVÁNÍ ---
# Všechny výpisy půjdou do standardního výstupu, který zachytí systemd
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])


# --- KONFIGURACE BOTA (NAČÍTÁNÍ Z .ENV) ---
load_dotenv()
API_KEY = os.getenv("DEMO_API_KEY")
API_SECRET = os.getenv("DEMO_API_SECRET")
if not API_KEY or not API_SECRET:
    logging.critical("CHYBA: API_KEY nebo API_SECRET nebyly nalezeny v souboru .env! Ukončuji program.")
    sys.exit()

# Ostatní konfigurační proměnné
SYMBOL = 'BTCUSDT'
TIMEFRAME = 15
RISK_PERCENT = 1.0
RISK_REWARD_RATIO = 2.0
ICHIMOKU_SETTINGS = {"tenkan": 9, "kijun": 26, "senkou": 52}

# --- PŘIPOJENÍ K BYBIT DEMO API ---
try:
    logging.info("Inicializuji připojení k Bybit Demo Trading...")
    session = HTTP(api_key=API_KEY, api_secret=API_SECRET, demo=True)
    session.get_kline(category='linear', symbol=SYMBOL, interval=TIMEFRAME, limit=1)
    logging.info("Připojení k Bybit API úspěšné.")
except Exception as e:
    logging.critical(f"Kritická chyba při připojování k Bybit API: {e}. Ukončuji program.")
    sys.exit()

# --- DEFINICE FUNKCÍ ---
def log_trade_to_sheet(trade_data_dict):
    try:
        headers = logbook_sheet.row_values(1)
        ordered_values = [trade_data_dict.get(h, '') for h in headers]
        logbook_sheet.append_row(ordered_values)
        logging.info(f"Obchod úspěšně zapsán do Google Tabulky.")
    except Exception as e:
        logging.error(f"Nepodařilo se zapsat obchod do Google Tabulky: {e}")

def get_current_balance():
    try:
        balance_data = session.get_wallet_balance(accountType="UNIFIED")
        if balance_data.get('retCode') == 0:
            coin_list = balance_data.get('result', {}).get('list', [{}])[0].get('coin', [])
            for coin in coin_list:
                if coin.get('coin') == 'USDT':
                    return float(coin.get('walletBalance', 0))
    except Exception as e:
        logging.error(f"Nepodařilo se získat zůstatek účtu: {e}")
    return None

def get_open_position():
    try:
        positions = session.get_positions(category="linear", symbol=SYMBOL)
        if positions.get('retCode') == 0:
            position_list = positions.get('result', {}).get('list', [])
            if position_list and float(position_list[0].get('size', 0)) > 0:
                return position_list[0]
    except Exception as e:
        logging.error(f"Nepodařilo se zkontrolovat otevřené pozice: {e}")
    return None

# --- HLAVNÍ SMYČKA BOTA ---
def run_bot():
    logging.info("==================================================")
    logging.info("Obchodní bot spuštěn.")
    logging.info("==================================================")

    while True:
        try:
            open_position = get_open_position()
            if open_position:
                logging.info(f"Detekována otevřená pozice o velikosti {open_position['size']} {SYMBOL}. Čekám na uzavření...")
                time.sleep(900)
                continue

            logging.info(f"Načítám čerstvá data pro {SYMBOL}...")
            
            kline = session.get_kline(category='linear', symbol=SYMBOL, interval=TIMEFRAME, limit=200)
            ohlcv = kline['result']['list']
            df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['date'] = pd.to_datetime(df['date'].astype(int), unit='ms')
            df.set_index('date', inplace=True)
            df = df.astype(float).sort_index()

            df.ta.ichimoku(tenkan=ICHIMOKU_SETTINGS["tenkan"], kijun=ICHIMOKU_SETTINGS["kijun"],
                           senkou=ICHIMOKU_SETTINGS["senkou"], append=True)

            new_column_names = {}
            for col in df.columns:
                if col.startswith(('ISA', 'ISB', 'ITS', 'IKS', 'ICS')):
                    new_column_names[col] = col.split('_')[0].lower()
            df.rename(columns=new_column_names, inplace=True)
            df.dropna(inplace=True)

            if len(df) < 3:
                logging.info("Nedostatek dat po výpočtu indikátorů. Čekám na další svíčku.")
                time.sleep(900)
                continue
            
            last_candle = df.iloc[-2]
            previous_candle = df.iloc[-3]

            is_long_signal = (last_candle['close'] > last_candle['isa'] and
                              last_candle['close'] > last_candle['isb'] and
                              last_candle['ics'] > last_candle['high'] and
                              previous_candle['its'] <= previous_candle['iks'] and
                              last_candle['its'] > last_candle['iks'])

            is_short_signal = (last_candle['close'] < last_candle['isa'] and
                               last_candle['close'] < last_candle['isb'] and
                               last_candle['ics'] < last_candle['low'] and
                               previous_candle['its'] >= previous_candle['iks'] and
                               last_candle['its'] < last_candle['iks'])

            if is_long_signal or is_short_signal:
                position_type = "LONG" if is_long_signal else "SHORT"
                entry_price = last_candle['close']
                stop_loss = last_candle['iks']

                if position_type == "LONG":
                    risk_per_unit = entry_price - stop_loss
                    take_profit = entry_price + (risk_per_unit * RISK_REWARD_RATIO)
                else:
                    risk_per_unit = stop_loss - entry_price
                    take_profit = entry_price - (risk_per_unit * RISK_REWARD_RATIO)

                if risk_per_unit <= 0:
                    logging.info(f"Signál detekován, ale riziko je nulové nebo záporné ({risk_per_unit:.2f}). Přeskakuji obchod.")
                    time.sleep(900)
                    continue

                account_balance = get_current_balance()
                if account_balance is None or account_balance == 0:
                    logging.error("Nelze pokračovat, zůstatek účtu je 0 nebo nedostupný.")
                    time.sleep(900)
                    continue

                logging.info(f"!!! {position_type} SIGNÁL na ceně {entry_price:.2f} !!!")
                logging.info(f"Aktuální zůstatek účtu: {account_balance:.2f} USDT")

                max_risk_usd = account_balance * (RISK_PERCENT / 100)
                position_size = round(max_risk_usd / risk_per_unit, 3)

                if position_size < 0.001:
                    logging.warning(f"Vypočtená velikost pozice ({position_size}) je příliš malá. Přeskakuji obchod.")
                    time.sleep(900)
                    continue

                logging.info(f"-> Vstupuji do {position_type} pozice o velikosti {position_size} {SYMBOL}")
                logging.info(f"-> SL: {stop_loss:.2f}, TP: {take_profit:.2f}")

                order = session.place_order(
                    category="linear", symbol=SYMBOL,
                    side="Buy" if position_type == "LONG" else "Sell",
                    orderType="Market", qty=str(position_size),
                    takeProfit=str(round(take_profit, 2)),
                    stopLoss=str(round(stop_loss, 2)),
                )

                if order.get('retCode') == 0:
                    order_id = order['result']['orderId']
                    logging.info(f"!!! PŘÍKAZ K {position_type} ODESLÁN, ID: {order_id} !!!")

                    trade_log_data = {
                        "timestamp": datetime.now().strftime("%Y-m-%d %H:%M:%S"),
                        "symbol": SYMBOL, "type": position_type, "size": position_size,
                        "entry_price": entry_price, "stop_loss": stop_loss,
                        "take_profit": take_profit, "order_id": order_id,
                        "risk_usd": max_risk_usd, "account_balance_before": account_balance
                    }
                    log_trade_to_sheet(trade_log_data)
                    
                    time.sleep(10)
                else:
                    logging.error(f"Chyba při odesílání příkazu: {order.get('retMsg')}")
            else:
                logging.info("Žádný signál, pokračuji v monitorování...")

            time.sleep(900)

        except Exception as e:
            logging.error(f"Nastala neočekávaná chyba v hlavní smyčce: {e}", exc_info=True)
            time.sleep(900)

if __name__ == '__main__':
    run_bot()