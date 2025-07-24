import asyncio
from datetime import datetime
import pandas as pd
import yfinance as yf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    ContextTypes, CallbackContext
)
import math
from scipy.stats import norm

import matplotlib.pyplot as plt
import io


import numpy as np

# === Config ===
REFRESH_INTERVAL = 60
user_monitor_tasks = {}  # {user_id: { (ticker, expiration): asyncio.Task }}
user_filter_settings = {}  # {user_id: {ticker: filters}}

DEFAULT_FILTERS = {
    "vol": 75,
    "premium": 50_000_000,
    "ratio": 1.0,
    "min_price": 0.3
}

# === Utilities ===
def fetch_options_data(ticker_symbol, expiration_date):
    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(expiration_date)
    calls, puts = chain.calls.copy(), chain.puts.copy()
    calls["type"] = "call"
    puts["type"] = "put"
    combined = pd.concat([calls, puts], ignore_index=True)
    combined["expiration"] = expiration_date
    return combined[["type", "strike", "lastPrice", "volume", "openInterest", "impliedVolatility"]]

def calculate_delta(option_type, stock_price, strike_price, time_to_expiry, iv):
    if iv <= 0 or time_to_expiry <= 0 or stock_price <= 0 or strike_price <= 0:
        return None
    d1 = (math.log(stock_price / strike_price) + (0.5 * iv ** 2) * time_to_expiry) / (iv * math.sqrt(time_to_expiry))
    return norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)

def compute_suggested_filters(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        expirations = yf_ticker.options[:3]  # Only look at 3 near-term expirations
        options = []

        for exp in expirations:
            chain = yf_ticker.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["type"] = "call"
            puts["type"] = "put"
            combined = pd.concat([calls, puts], ignore_index=True)
            combined["expiration"] = exp
            options.append(combined)

        all_options = pd.concat(options, ignore_index=True)
        all_options.dropna(subset=["volume", "lastPrice", "openInterest"], inplace=True)
        all_options = all_options[all_options["volume"] > 0]

        all_options["premium"] = all_options["volume"] * all_options["lastPrice"] * 100
        all_options["vol_oi"] = all_options.apply(
            lambda row: row["volume"] / row["openInterest"] if row["openInterest"] > 0 else 0, axis=1
        )

        filters = {
            "vol": int(max(all_options["volume"].quantile(0.75), 100)),
            "premium": int(max(all_options["premium"].quantile(0.75), 100_000)),
            "ratio": round(max(all_options["vol_oi"].median(), 1.0), 2),
            "min_price": round(max(all_options["lastPrice"].quantile(0.25), 0.5), 2)
        }

        return filters

    except Exception as e:
        print(f"Autofilter error for {ticker}: {e}")
        return DEFAULT_FILTERS

def analyze_option_changes(ticker, expiration, prev_df, curr_df, stock_price, filters):
    alerts = []
    for _, row in curr_df.iterrows():
        opt_type, strike, last_price = row["type"], row["strike"], row["lastPrice"]
        volume, open_interest, iv = row["volume"], row["openInterest"], row.get("impliedVolatility", None)

        if pd.isna(volume) or pd.isna(last_price) or last_price < filters["min_price"]:
            continue

        prev_row = None if prev_df is None else prev_df[
            (prev_df["strike"] == strike) & (prev_df["type"] == opt_type)
        ]

        if prev_row is not None and not prev_row.empty:
            prev_volume = prev_row["volume"].values[0]
            volume_change = volume - prev_volume
            if volume_change <= 0:
                continue

            premium = last_price * volume_change * 100
            vol_oi_ratio = volume / open_interest if open_interest > 0 else 0
            time_to_expiry = max((pd.to_datetime(expiration) - pd.Timestamp.now()).days / 365, 0.01)
            delta = calculate_delta(opt_type, stock_price, strike, time_to_expiry, iv) if iv is not None else None

            if premium < filters["premium"] or volume_change < filters["vol"] or vol_oi_ratio < filters["ratio"]:
                continue

            label = "C" if opt_type == "call" else "P"
            role = "INSTITUTIONAL BUY" if opt_type == "call" else "HEDGE/SELL"
            premium_pct = (premium / (stock_price * volume_change * 100)) * 100 if stock_price > 0 else 0

            message = (
                f"\n📈 {ticker} ${strike:.2f}{label} ({expiration})\n"
                f"──────────────────────────────\n"
                f"{'🟢' if opt_type == 'call' else '🔴'} {role}\n"
                f"💵 Premium: ${premium:,.0f} ({premium_pct:.2f}%)\n"
                f"📊 Volume: +{volume_change:.1f} | Total: {volume:,.1f}\n"
                f"💰 Price: ${last_price:.2f}  | Stock: ${stock_price:.2f}\n"
                f"📈 Vol/OI Ratio: {vol_oi_ratio:.2f}\n"
                f"📉 IV: {'N/A' if iv is None else f'{iv:.2%}'} | Δ: {'N/A' if delta is None else f'{delta:.2f}'}\n"
                f"⏱️ {datetime.now().strftime('%H:%M:%S')}"
            )
            alerts.append(message)
    return alerts

async def monitor_ticker(ticker, expiration, user_id, context: ContextTypes.DEFAULT_TYPE):
    prev_data = None
    chat_id = user_id
    await context.bot.send_message(chat_id=chat_id, text=f"✅ Monitoring {ticker} ({expiration}) every {REFRESH_INTERVAL}s.")

    while True:
        try:
            current_data = fetch_options_data(ticker, expiration)
            stock_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]

            filters = DEFAULT_FILTERS.copy()
            user_filters = user_filter_settings.get(user_id, {})
            if isinstance(user_filters, dict):
                filters.update(user_filters.get(ticker, {}))

            alerts = analyze_option_changes(ticker, expiration, prev_data, current_data, stock_price, filters)
            prev_data = current_data

            for alert in alerts:
                await context.bot.send_message(chat_id=chat_id, text=alert)

            await asyncio.sleep(REFRESH_INTERVAL)

        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Error: {e}")
            await asyncio.sleep(REFRESH_INTERVAL)

# === Commands ===
async def monitor_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /monitor TICKER")
        return
    ticker = args[0].upper()
    ticker_obj = yf.Ticker(ticker)
    expirations = ticker_obj.options
    if not expirations:
        await update.message.reply_text(f"No expiration dates found for {ticker}.")
        return
    buttons = [[InlineKeyboardButton(date, callback_data=f"toggle_exp|{ticker}|{date}")] for date in expirations]
    buttons.append([InlineKeyboardButton("✅ Start Monitoring Selected", callback_data=f"start_monitor_multi|{ticker}")])
    context.user_data["selected_expirations"] = set()
    await update.message.reply_text(
        f"📆 Select expiration dates for {ticker} (toggle buttons):",
        reply_markup=InlineKeyboardMarkup(buttons)
    )

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_monitor_tasks or not user_monitor_tasks[user_id]:
        await update.message.reply_text("You have no active monitors.")
        return
    buttons = [[InlineKeyboardButton(f"{ticker} ({exp})", callback_data=f"stop_monitor|{ticker}|{exp}")]
               for (ticker, exp) in user_monitor_tasks[user_id]]
    buttons.append([InlineKeyboardButton("🛑 Stop All", callback_data="stop_monitor|ALL|ALL")])
    await update.message.reply_text("Select a monitor to stop:", reply_markup=InlineKeyboardMarkup(buttons))

async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_monitor_tasks or not user_monitor_tasks[user_id]:
        await update.message.reply_text("📭 No active monitors.")
        return
    active = "\n".join([f"• {t} ({d})" for (t, d) in user_monitor_tasks[user_id]])
    await update.message.reply_text(f"📊 Active Monitors:\n{active}")

async def setfilters_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    try:
        if len(args) != 5:
            raise ValueError()
        ticker = args[0].upper()
        user_filter_settings.setdefault(user_id, {})[ticker] = {
            "vol": int(args[1]),
            "premium": int(args[2]),
            "ratio": float(args[3]),
            "min_price": float(args[4]),
        }
        await update.message.reply_text(f"✅ Filters updated for {ticker}!")
    except:
        await update.message.reply_text("Usage: /setfilters TICKER volume premium ratio min_price\nExample: /setfilters AAPL 500 250000 2.0 1.0")

async def getfilters_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    ticker = args[0].upper() if args else None
    user_filters = user_filter_settings.get(user_id, {})
    filters = user_filters.get(ticker, DEFAULT_FILTERS) if ticker else DEFAULT_FILTERS
    await update.message.reply_text(
        f"📋 Filters for {ticker or 'DEFAULT'}:\n"
        f"• Volume ≥ {filters['vol']}\n"
        f"• Premium ≥ ${filters['premium']:,.0f}\n"
        f"• Volume/Open Interest ≥ {filters['ratio']:.2f}\n"
        f"• Option Price ≥ ${filters['min_price']:.2f}"
    )

async def autofilter_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /autofilter TICKER")
        return
    ticker = args[0].upper()
    filters = compute_suggested_filters(ticker)
    if not filters:
        await update.message.reply_text("⚠️ Could not calculate filters for this ticker.")
        return
    user_filter_settings.setdefault(user_id, {})[ticker] = filters
    await update.message.reply_text(
        f"✅ Suggested filters applied for {ticker}:\n"
        f"• Volume ≥ {filters['vol']}\n"
        f"• Premium ≥ ${filters['premium']:,.0f}\n"
        f"• Vol/OI Ratio ≥ {filters['ratio']}\n"
        f"• Option Price ≥ ${filters['min_price']:.2f}"
    )

async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /summary TICKER")
        return

    ticker = args[0].upper()
    try:
        yf_ticker = yf.Ticker(ticker)
        stock_price = yf_ticker.history(period="1d")["Close"].iloc[-1]
        expirations = yf_ticker.options[:3]

        all_data = []
        for exp in expirations:
            chain = yf_ticker.option_chain(exp)
            for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
                df = df.copy()
                df["type"] = opt_type
                df["expiration"] = exp
                df["premium"] = df["lastPrice"] * df["volume"] * 100
                all_data.append(df)

        options_df = pd.concat(all_data)
        options_df.dropna(subset=["strike", "volume", "premium"], inplace=True)

        # === Summary Text ===
        total_vol = int(options_df["volume"].sum())
        total_prem = int(options_df["premium"].sum())
        top_strike = options_df.groupby("strike")["volume"].sum().idxmax()
        pcr = (
            options_df[options_df["type"] == "put"]["volume"].sum() /
            max(1, options_df[options_df["type"] == "call"]["volume"].sum())
        )
        summary_text = (
            f"📊 Summary for {ticker} (Price: ${stock_price:.2f})\n"
            f"────────────────────────────\n"
            f"🔹 Total Option Volume: {total_vol:,}\n"
            f"🔸 Total Premium: ${total_prem:,}\n"
            f"💥 Most Active Strike: ${top_strike:.2f}\n"
            f"📈 Put/Call Volume Ratio: {pcr:.2f}\n"
            f"📆 Expirations analyzed: {', '.join(expirations)}"
        )
        await context.bot.send_message(chat_id=user_id, text=summary_text)

        # === Volume by Strike (Binned) ===
        options_df["strike_bucket"] = options_df["strike"].round(-1)
        grouped = options_df.groupby(["strike_bucket", "type"])["volume"].sum().unstack().fillna(0)

        fig, ax = plt.subplots(figsize=(14, 6))
        grouped.plot(kind='bar', stacked=False, ax=ax, alpha=0.7, width=0.8, color={'call': 'skyblue', 'put': 'salmon'})
        ax.axvline((stock_price // 10), color="black", linestyle="--", linewidth=1.5, label="ATM Price Bucket")
        ax.annotate("← ATM", xy=((stock_price // 10), grouped.max().max()*0.9), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="center", color="black")

        ax.set_title(f"{ticker} Option Volume by Strike Bucket")
        ax.set_xlabel("Strike Price Bucket")
        ax.set_ylabel("Volume")
        ax.legend()
        plt.tight_layout()

        buf1 = io.BytesIO()
        plt.savefig(buf1, format="png")
        buf1.seek(0)
        await context.bot.send_photo(chat_id=user_id, photo=buf1)
        plt.clf()

        # === Premium Heatmap ===
        pivot = options_df.pivot_table(index="strike", columns="expiration", values="premium", aggfunc="sum")
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        im = ax2.imshow(pivot.fillna(0), cmap="viridis", aspect="auto", interpolation="nearest", origin="lower")

        # Downsample ticks for readability
        xticks = list(range(0, len(pivot.columns), max(1, len(pivot.columns)//10)))
        yticks = list(range(0, len(pivot.index), max(1, len(pivot.index)//20)))

        ax2.set_xticks(xticks)
        ax2.set_xticklabels([pivot.columns[i] for i in xticks], rotation=45, ha="right")
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([f"{pivot.index[i]:.0f}" for i in yticks])

        ax2.set_title(f"{ticker} Premium Heatmap")
        ax2.set_xlabel("Expiration Date")
        ax2.set_ylabel("Strike Price")

        # Highlight ATM row
        if stock_price in pivot.index:
            atm_idx = pivot.index.get_loc(stock_price)
        else:
            closest_strike = pivot.index[np.abs(pivot.index - stock_price).argmin()]
            atm_idx = pivot.index.get_loc(closest_strike)
        ax2.axhline(y=atm_idx, color="white", linestyle="--", linewidth=1.5)

        fig2.colorbar(im, ax=ax2)
        plt.tight_layout()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png")
        buf2.seek(0)
        await context.bot.send_photo(chat_id=user_id, photo=buf2)
        plt.clf()

    except Exception as e:
        await update.message.reply_text(f"⚠️ Error generating summary: {e}")



async def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data
    if data.startswith("toggle_exp"):
        _, ticker, expiration = data.split("|")
        selected = context.user_data.get("selected_expirations", set())
        selected ^= {expiration}
        context.user_data["selected_expirations"] = selected
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options
        buttons = [[InlineKeyboardButton(f"{'✅ ' if date in selected else ''}{date}", callback_data=f"toggle_exp|{ticker}|{date}")]
                   for date in expirations]
        buttons.append([InlineKeyboardButton("✅ Start Monitoring Selected", callback_data=f"start_monitor_multi|{ticker}")])
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(buttons))
    elif data.startswith("start_monitor_multi"):
        _, ticker = data.split("|")
        selected = context.user_data.get("selected_expirations", set())
        if not selected:
            await query.edit_message_text("⚠️ Please select at least one expiration date.")
            return
        user_monitor_tasks.setdefault(user_id, {})
        started = []
        for expiration in selected:
            key = (ticker, expiration)
            if key not in user_monitor_tasks[user_id]:
                task = asyncio.create_task(monitor_ticker(ticker, expiration, user_id, context))
                user_monitor_tasks[user_id][key] = task
                started.append(expiration)
        await query.edit_message_text(f"📡 Started monitoring {ticker}:\n" + "\n".join(started))
        context.user_data["selected_expirations"] = set()
    elif data.startswith("stop_monitor"):
        _, ticker, expiration = data.split("|")
        if ticker == "ALL":
            for task in user_monitor_tasks.get(user_id, {}).values():
                task.cancel()
            user_monitor_tasks[user_id] = {}
            await query.edit_message_text("🛑 Stopped all monitors.")
        else:
            key = (ticker, expiration)
            task = user_monitor_tasks[user_id].pop(key, None)
            if task:
                task.cancel()
                await query.edit_message_text(f"🛑 Stopped {ticker} ({expiration}).")
            else:
                await query.edit_message_text("Monitor not found.")

# === Main ===
if __name__ == "__main__":
    TOKEN = "7518072183:AAH6NxpeN1H2Wwc0Th1oC0A19A1qF1a5Sww"
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("summary", summary_command))
    app.add_handler(CommandHandler("monitor", monitor_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("list", list_command))
    app.add_handler(CommandHandler("setfilters", setfilters_command))
    app.add_handler(CommandHandler("getfilters", getfilters_command))
    app.add_handler(CommandHandler("autofilter", autofilter_command))
    app.add_handler(CallbackQueryHandler(button_handler))

    print("Bot running...")
    app.run_polling()