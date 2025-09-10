# Tento skript provádí historický test (backtest) obchodní strategie založené na
# indikátoru Ichimoku Cloud. Testuje tři různé výstupní strategie na zadaném
# časovém rámci (timeframe). Zahrnuje money management s fixním procentuálním
# riskem na každý obchod. Výstupem je souhrnná analýza výkonu a detailní
# logbooky obchodů uložené v CSV souborech.


import pandas as pd
import pandas_ta as ta
import numpy as np
import os

# --- KONFIGURACE ---
DATA_DIRECTORY = ''
INITIAL_CAPITAL = 1000.0  # Počáteční kapitál v USDT
RISK_PERCENT = 1.0       # Risk na jeden obchod v procentech

ICHIMOKU_SETTINGS = {
    "tenkan": 9,
    "kijun": 26,
    "senkou": 52
}
RISK_REWARD_RATIO = 2.0

# --- HLAVNÍ FUNKCE ---

def run_backtest(df, exit_strategy, initial_capital=1000.0, risk_percent=1.0):
    trades_log = []
    in_position = False
    position_type = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_index = 0
    position_size = 0
    
    account_balance = initial_capital

    for i in range(1, len(df)):
        if in_position:
            exit_price = None
            reason = None
            if (position_type == 'LONG' and df['low'].iloc[i] <= stop_loss) or \
               (position_type == 'SHORT' and df['high'].iloc[i] >= stop_loss):
                exit_price, reason = stop_loss, 'Stop-Loss'
            elif exit_strategy == 'A' and \
                ((position_type == 'LONG' and df['high'].iloc[i] >= take_profit) or \
                 (position_type == 'SHORT' and df['low'].iloc[i] <= take_profit)):
                exit_price, reason = take_profit, 'Take-Profit'
            elif exit_strategy == 'B' and \
                 ((position_type == 'LONG' and df['tenkan'].iloc[i] < df['kijun'].iloc[i]) or \
                  (position_type == 'SHORT' and df['tenkan'].iloc[i] > df['kijun'].iloc[i])):
                exit_price, reason = df['close'].iloc[i], 'Opposite Signal'
            
            if exit_price is not None:
                pnl = ((exit_price - entry_price) * position_size) if position_type == 'LONG' else ((entry_price - exit_price) * position_size)
                account_balance += pnl
                
                trades_log.append({
                    'entry_date': df.index[entry_index], 'exit_date': df.index[i], 'type': position_type,
                    'entry_price': entry_price, 'exit_price': exit_price, 'position_size': position_size,
                    'pnl': pnl, 'reason': reason, 'account_balance': account_balance
                })
                in_position = False
                
            elif exit_strategy == 'C':
                if position_type == 'LONG': stop_loss = max(stop_loss, df['kijun'].iloc[i])
                elif position_type == 'SHORT': stop_loss = min(stop_loss, df['kijun'].iloc[i])
        
        if not in_position:
            is_long_signal = df['close'].iloc[i] > df['senkou_a'].iloc[i] and \
                             df['close'].iloc[i] > df['senkou_b'].iloc[i] and \
                             df['chikou'].iloc[i] > df['high'].iloc[i] and \
                             df['tenkan'].iloc[i-1] <= df['kijun'].iloc[i-1] and \
                             df['tenkan'].iloc[i] > df['kijun'].iloc[i]

            is_short_signal = df['close'].iloc[i] < df['senkou_a'].iloc[i] and \
                              df['close'].iloc[i] < df['senkou_b'].iloc[i] and \
                              df['chikou'].iloc[i] < df['low'].iloc[i] and \
                              df['tenkan'].iloc[i-1] >= df['kijun'].iloc[i-1] and \
                              df['tenkan'].iloc[i] < df['kijun'].iloc[i]

            if is_long_signal:
                entry_price = df['close'].iloc[i]
                if exit_strategy == 'A': stop_loss = df['kijun'].iloc[i]
                elif exit_strategy == 'B': stop_loss = min(df['senkou_a'].iloc[i], df['senkou_b'].iloc[i])
                elif exit_strategy == 'C': stop_loss = df['kijun'].iloc[i]
                
                # --- VÝPOČET VELIKOSTI POZICE ---
                risk_per_unit = entry_price - stop_loss
                if risk_per_unit <= 0: continue
                max_risk_usd = account_balance * (risk_percent / 100)
                position_size = max_risk_usd / risk_per_unit
                # --- KONEC VÝPOČTU ---
                
                in_position, position_type, entry_index = True, 'LONG', i
                if exit_strategy == 'A': take_profit = entry_price + (risk_per_unit * RISK_REWARD_RATIO)

            elif is_short_signal:
                entry_price = df['close'].iloc[i]
                if exit_strategy == 'A': stop_loss = df['kijun'].iloc[i]
                elif exit_strategy == 'B': stop_loss = max(df['senkou_a'].iloc[i], df['senkou_b'].iloc[i])
                elif exit_strategy == 'C': stop_loss = df['kijun'].iloc[i]

                # --- VÝPOČET VELIKOSTI POZICE ---
                risk_per_unit = stop_loss - entry_price
                if risk_per_unit <= 0: continue
                max_risk_usd = account_balance * (risk_percent / 100)
                position_size = max_risk_usd / risk_per_unit
                # --- KONEC VÝPOČTU ---

                in_position, position_type, entry_index = True, 'SHORT', i
                if exit_strategy == 'A': take_profit = entry_price - (risk_per_unit * RISK_REWARD_RATIO)
                
    return pd.DataFrame(trades_log)

def analyze_results(trades_df, initial_capital=1000.0):
    if trades_df.empty:
        print("\n--- Analýza výkonu ---")
        print("Nebyly provedeny žádné obchody.")
        return
        
    final_balance = trades_df['account_balance'].iloc[-1]
    total_pnl = final_balance - initial_capital
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = winning_trades['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    trades_df['peak'] = trades_df['account_balance'].cummax()
    trades_df['drawdown'] = trades_df['peak'] - trades_df['account_balance']
    max_drawdown = trades_df['drawdown'].max()
    max_drawdown_percent = (max_drawdown / trades_df['peak'].max()) * 100 if trades_df['peak'].max() > 0 else 0

    print("\n--- Analýza výkonu ---")
    print(f"Počáteční kapitál: {initial_capital:.2f} USDT")
    print(f"Konečný kapitál: {final_balance:.2f} USDT")
    print(f"Celkový zisk/ztráta (PnL): {total_pnl:.2f} USDT")
    print(f"Celkový počet obchodů: {total_trades}")
    print(f"Úspěšnost (Win Rate): {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Maximální Drawdown: {max_drawdown:.2f} USDT ({max_drawdown_percent:.2f}%)")

def main():
    logbook_dir = 'logbooks_1_perc_risk' # Nová složka pro nové výsledky
    os.makedirs(logbook_dir, exist_ok=True)

    timeframe = '15M'
    filename = 'BTCUSDT_15m.csv'
    filepath = os.path.join(DATA_DIRECTORY, filename)

    print(f"\n==========================================================")
    print(f"ZAČÍNÁM BACKTEST PRO: BTC/USDT, Timeframe: {timeframe.upper()}")
    print(f"Risk na obchod: {RISK_PERCENT}% | Počáteční kapitál: {INITIAL_CAPITAL} USDT")
    print(f"==========================================================")

    if not os.path.exists(filepath):
        print(f"Chyba: Soubor {filename} nebyl nalezen."); return

    try:
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = df.columns.str.lower()
        if 'date' not in df.columns: print(f"Chyba: Chybí sloupec 'date'."); return
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    except Exception as e:
        print(f"Chyba při načítání souboru {filename}: {e}"); return
    
    original_columns = set(df.columns)
    df.ta.ichimoku(tenkan=ICHIMOKU_SETTINGS["tenkan"], kijun=ICHIMOKU_SETTINGS["kijun"], senkou=ICHIMOKU_SETTINGS["senkou"], append=True)
    new_columns = set(df.columns)
    added_columns = new_columns - original_columns
    
    rename_map = {}
    for col in added_columns:
        if col.startswith('ITS'): rename_map[col] = 'tenkan'
        elif col.startswith('IKS'): rename_map[col] = 'kijun'
        elif col.startswith('ISA'): rename_map[col] = 'senkou_a'
        elif col.startswith('ISB'): rename_map[col] = 'senkou_b'
        elif col.startswith('ICS'): rename_map[col] = 'chikou'
    df.rename(columns=rename_map, inplace=True)
    
    if not all(k in df.columns for k in ['tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou']):
        print("Chyba: Nepodařilo se vytvořit všechny Ichimoku sloupce."); return

    df.dropna(inplace=True)

    exit_strategies = {'A: Pevný R:R poměr (1:2)': 'A', 'B: Výstup na opačný signál': 'B', 'C: Trailing Stop na Kijun-sen': 'C'}
    
    for name, code in exit_strategies.items():
        print(f"\n--- Spouštím test pro výstupní strategii: {name} ---")
        logbook = run_backtest(df.copy(), code, initial_capital=INITIAL_CAPITAL, risk_percent=RISK_PERCENT)
        
        analyze_results(logbook, initial_capital=INITIAL_CAPITAL)
        
        if not logbook.empty:
            logbook_filename = f"logbook_{timeframe}_{code}_1_perc_risk.csv"
            logbook_filepath = os.path.join(logbook_dir, logbook_filename)
            logbook.to_csv(logbook_filepath, index=False)
            print(f"Detailní logbook byl uložen do: ./{logbook_filepath}")
        
        print("---------------------------")


if __name__ == "__main__":
    main()
