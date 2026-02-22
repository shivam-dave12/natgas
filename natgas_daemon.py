#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         NATURAL GAS PREDICTOR — CONTINUOUS DAEMON + TELEGRAM BOT          ║
║                                                                            ║
║  Wraps natgas_predictor_enhanced_v18.py and runs it continuously:          ║
║    • Auto-refreshes on NWP weather cycles (00Z/06Z/12Z/18Z + delay)       ║
║    • EIA storage report alert (Thursday 10:30 ET)                          ║
║    • Telegram bot: rich notifications + interactive commands               ║
║                                                                            ║
║  .env Configuration:                                                       ║
║    EIA_API_KEY=your_eia_key                                                ║
║    TELEGRAM_BOT_TOKEN=your_bot_token                                       ║
║    TELEGRAM_CHAT_ID=your_chat_id_or_channel_id                             ║
║                                                                            ║
║  Usage:                                                                    ║
║    python natgas_daemon.py                                                 ║
║    python natgas_daemon.py --no-telegram   # run without Telegram          ║
║    python natgas_daemon.py --run-once      # single run, notify, exit      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal
import logging
import traceback
import threading
import importlib
import importlib.util
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT,
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler("natgas_daemon.log", encoding="utf-8"),
                    ])
log = logging.getLogger("daemon")


# ═══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC IMPORT OF PREDICTOR MODULE
# ═══════════════════════════════════════════════════════════════════════════════

def _load_predictor_module():
    """
    Import the v18 predictor as a module.
    Searches for the file in the same directory as this daemon script.
    Handles filenames with 'v18' or 'enhanced' in the name.
    """
    script_dir = Path(__file__).parent
    candidates = [
        "natural_gas_predictor_enhanced_v18.py",
        "natural_gas_predictor_enhanced.py",
        "natgas_predictor_v18.py",
    ]
    for name in candidates:
        path = script_dir / name
        if path.exists():
            spec = importlib.util.spec_from_file_location("natgas_predictor", str(path))
            mod = importlib.util.module_from_spec(spec)
            # Suppress the if __name__ == "__main__" execution
            sys.modules["natgas_predictor"] = mod
            spec.loader.exec_module(mod)
            log.info(f"Loaded predictor module from {path.name}")
            return mod
    raise FileNotFoundError(
        f"Cannot find predictor script in {script_dir}. "
        f"Expected one of: {candidates}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM BOT
# ═══════════════════════════════════════════════════════════════════════════════

class TelegramBot:
    """
    Lightweight Telegram bot using raw HTTP requests.
    Supports: sending messages (HTML), receiving commands via getUpdates polling.
    No heavy async framework needed.
    """

    API = "https://api.telegram.org/bot{token}/{method}"
    MAX_MSG_LEN = 4000  # Telegram limit is 4096 — leave margin

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)
        self._last_update_id = 0
        self._command_handlers: Dict[str, callable] = {}

        if self.enabled:
            # Validate token by calling getMe
            try:
                me = self._call("getMe")
                bot_name = me.get("result", {}).get("username", "unknown")
                log.info(f"Telegram bot connected: @{bot_name}")
            except Exception as e:
                log.error(f"Telegram connection failed: {e}")
                self.enabled = False

    def _call(self, method: str, **params) -> dict:
        url = self.API.format(token=self.token, method=method)
        resp = requests.post(url, json=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def send(self, text: str, parse_mode: str = "HTML",
             disable_preview: bool = True, silent: bool = False) -> bool:
        """Send a message. Auto-splits if too long."""
        if not self.enabled:
            return False

        chunks = self._split_message(text)
        for chunk in chunks:
            try:
                self._call("sendMessage",
                           chat_id=self.chat_id,
                           text=chunk,
                           parse_mode=parse_mode,
                           disable_web_page_preview=disable_preview,
                           disable_notification=silent)
            except Exception as e:
                log.error(f"Telegram send failed: {e}")
                # Retry without parse_mode in case of formatting errors
                try:
                    self._call("sendMessage",
                               chat_id=self.chat_id,
                               text=chunk,
                               disable_web_page_preview=True)
                except Exception as e2:
                    log.error(f"Telegram send (plain) also failed: {e2}")
                    return False
        return True

    def _split_message(self, text: str) -> List[str]:
        if len(text) <= self.MAX_MSG_LEN:
            return [text]
        chunks = []
        while text:
            if len(text) <= self.MAX_MSG_LEN:
                chunks.append(text)
                break
            # Find a good split point (newline near the limit)
            split_at = text.rfind("\n", 0, self.MAX_MSG_LEN)
            if split_at < self.MAX_MSG_LEN // 2:
                split_at = self.MAX_MSG_LEN
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks

    def register(self, command: str, handler: callable, description: str = ""):
        """Register a /command handler."""
        self._command_handlers[command.lower().strip("/")] = {
            "fn": handler,
            "desc": description,
        }

    def poll_commands(self, timeout: int = 1) -> List[dict]:
        """Poll for new commands. Non-blocking (short timeout)."""
        if not self.enabled:
            return []
        try:
            result = self._call("getUpdates",
                                offset=self._last_update_id + 1,
                                timeout=timeout,
                                allowed_updates=["message"])
            updates = result.get("result", [])
            commands = []
            for upd in updates:
                self._last_update_id = upd["update_id"]
                msg = upd.get("message", {})
                text = msg.get("text", "").strip()
                chat_id = str(msg.get("chat", {}).get("id", ""))
                if text.startswith("/") and chat_id == str(self.chat_id):
                    parts = text.split(maxsplit=1)
                    cmd = parts[0].lower().strip("/").split("@")[0]  # handle /cmd@botname
                    args = parts[1] if len(parts) > 1 else ""
                    commands.append({"cmd": cmd, "args": args, "chat_id": chat_id})
            return commands
        except Exception as e:
            log.debug(f"Poll error (usually timeout): {e}")
            return []

    def process_commands(self, runner: "DaemonRunner"):
        """Check for commands and dispatch to handlers."""
        commands = self.poll_commands(timeout=2)
        for cmd_info in commands:
            cmd = cmd_info["cmd"]
            args = cmd_info["args"]
            handler_info = self._command_handlers.get(cmd)
            if handler_info:
                try:
                    handler_info["fn"](runner, args)
                except Exception as e:
                    log.error(f"Command /{cmd} error: {e}")
                    self.send(f"❌ Error executing /{cmd}: {_esc(str(e))}")
            else:
                self.send(f"❓ Unknown command: /{_esc(cmd)}\nUse /help to see available commands.")


# ═══════════════════════════════════════════════════════════════════════════════
#  NWP CYCLE SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class NWPScheduler:
    """
    Tracks NWP (Numerical Weather Prediction) model cycles and determines
    when to refresh the predictor.

    NWP cycles (UTC): 00Z, 06Z, 12Z, 18Z
    Data availability: typically 3-4 hours after cycle → check at +3.5h

    Also tracks:
    - EIA storage report: Thursday 10:30 ET (15:30 UTC in winter, 14:30 summer)
    - Market open/close for session awareness
    """

    # NWP cycle starts (UTC hours)
    NWP_CYCLES = [0, 6, 12, 18]
    # Delay after cycle start before data is usually available (hours)
    NWP_DELAY_H = 3.5
    # Minimum interval between runs (minutes) to prevent hammering
    MIN_INTERVAL_MIN = 30
    # EIA storage report: Thursday 15:30 UTC (10:30 ET winter)
    EIA_WEEKDAY = 3  # Thursday
    EIA_HOUR_UTC = 15
    EIA_MINUTE_UTC = 35  # 5 min after release to let data settle

    def __init__(self):
        self.last_run_utc: Optional[datetime] = None
        self.last_nwp_cycle: Optional[str] = None
        self.run_count: int = 0
        self.start_time = datetime.now(timezone.utc)

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def next_refresh_utc(self) -> Tuple[datetime, str]:
        """
        Returns (next_refresh_time_utc, reason).
        Reasons: 'nwp_00z', 'nwp_06z', 'nwp_12z', 'nwp_18z', 'eia_storage', 'startup'
        """
        now = self._utc_now()

        if self.last_run_utc is None:
            return now, "startup"

        candidates = []

        # NWP cycles
        for cycle_h in self.NWP_CYCLES:
            check_h = cycle_h + self.NWP_DELAY_H
            check_time = now.replace(hour=int(check_h), minute=int((check_h % 1) * 60),
                                     second=0, microsecond=0)
            if check_time <= now:
                # Check if we already ran for this cycle today
                cycle_key = f"{now.date()}_{cycle_h:02d}z"
                if cycle_key != self.last_nwp_cycle:
                    candidates.append((check_time, f"nwp_{cycle_h:02d}z"))
            else:
                candidates.append((check_time, f"nwp_{cycle_h:02d}z"))

        # EIA storage report (Thursday)
        today = now.date()
        days_until_thu = (self.EIA_WEEKDAY - today.weekday()) % 7
        if days_until_thu == 0 and now.hour >= self.EIA_HOUR_UTC:
            # Already past this Thursday's report — next Thursday
            days_until_thu = 7
        next_thu = today + timedelta(days=days_until_thu)
        eia_time = datetime(next_thu.year, next_thu.month, next_thu.day,
                            self.EIA_HOUR_UTC, self.EIA_MINUTE_UTC,
                            tzinfo=timezone.utc)
        candidates.append((eia_time, "eia_storage"))

        # Filter: only future times or times we haven't run for yet
        future = [(t, r) for t, r in candidates if t > (self.last_run_utc or now)]
        if not future:
            # All cycles already handled — find next day's first cycle
            tomorrow = (now + timedelta(days=1)).date()
            first_cycle = datetime(tomorrow.year, tomorrow.month, tomorrow.day,
                                   int(self.NWP_DELAY_H),
                                   int((self.NWP_DELAY_H % 1) * 60),
                                   tzinfo=timezone.utc)
            return first_cycle, "nwp_00z"

        future.sort(key=lambda x: x[0])
        return future[0]

    def should_run_now(self) -> Tuple[bool, str]:
        """Check if we should run now. Returns (should_run, reason)."""
        now = self._utc_now()

        if self.last_run_utc is None:
            return True, "startup"

        # Enforce minimum interval
        elapsed = (now - self.last_run_utc).total_seconds() / 60
        if elapsed < self.MIN_INTERVAL_MIN:
            return False, f"cooldown ({elapsed:.0f}m < {self.MIN_INTERVAL_MIN}m)"

        next_time, reason = self.next_refresh_utc()
        if now >= next_time:
            return True, reason

        return False, f"waiting (next: {next_time.strftime('%H:%M')} UTC — {reason})"

    def mark_run(self, reason: str):
        now = self._utc_now()
        self.last_run_utc = now
        self.run_count += 1
        if reason.startswith("nwp_"):
            self.last_nwp_cycle = f"{now.date()}_{reason.replace('nwp_', '')}"
        log.info(f"Run #{self.run_count} completed ({reason})")


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML ESCAPE UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _esc(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _bold(text: str) -> str:
    return f"<b>{_esc(str(text))}</b>"


def _code(text: str) -> str:
    return f"<code>{_esc(str(text))}</code>"


def _pre(text: str) -> str:
    return f"<pre>{_esc(str(text))}</pre>"


def _mono_table(headers: List[str], rows: List[List[str]], alignments: str = "") -> str:
    """Build a monospace table string for Telegram <pre> blocks."""
    if not rows:
        return ""
    # Calculate column widths
    all_rows = [headers] + rows
    widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]
    lines = []
    # Header
    hdr = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    lines.append(hdr)
    lines.append("─" * len(hdr))
    # Rows
    for row in rows:
        line = "  ".join(str(c).ljust(w) for c, w in zip(row, widths))
        lines.append(line)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  MESSAGE FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_standby_msg() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "🔥 <b>NatGas Predictor v18 — Online</b>\n"
        f"📅 {_esc(now)}\n\n"
        "⏸️ Status: <b>STANDBY</b>\n"
        "Listening for commands. No model loaded yet.\n\n"
        "👉 Send /start to activate the model\n"
        "💬 Send /help for all commands"
    )


def fmt_startup_msg() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "🔥 <b>NatGas Predictor v18 — Daemon Started</b>\n"
        f"📅 {_esc(now)}\n\n"
        "✅ Model loading and first run starting...\n"
        "📡 Auto-refresh: every NWP cycle (00Z/06Z/12Z/18Z)\n"
        "💬 Send /help for commands"
    )


def fmt_run_complete(preds, predictor, reason: str, duration_s: float) -> str:
    """Format the main forecast notification."""
    if preds is None or preds.empty:
        return "⚠️ <b>Model run completed but produced no predictions</b>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ref_price = getattr(predictor, '_forecast_current', None) or predictor.live_price or predictor.current_price

    # Extract key info
    first = preds.iloc[0]
    spike_risk = first.get("spike_risk", 0)
    spike_regime = first.get("spike_regime", "NORMAL")
    predicted_close = first.get("predicted", 0)
    change_pct = first.get("change_pct", 0)
    low5 = first.get("low_5pct", 0)
    high95 = first.get("high_95pct", 0)

    # Direction emoji
    if change_pct > 1:
        dir_emoji = "🟢📈"
        dir_text = "BULLISH"
    elif change_pct < -1:
        dir_emoji = "🔴📉"
        dir_text = "BEARISH"
    else:
        dir_emoji = "⚪➡️"
        dir_text = "NEUTRAL"

    # Spike risk badge
    if spike_risk >= 0.7:
        risk_badge = "🔴 EXTREME"
    elif spike_risk >= 0.5:
        risk_badge = "🟠 HIGH"
    elif spike_risk >= 0.3:
        risk_badge = "🟡 ELEVATED"
    else:
        risk_badge = "🟢 LOW"

    # Short-term average (1-3d)
    short_preds = preds[preds["horizon"].between(0, 3)]
    short_avg = short_preds["predicted"].mean() if not short_preds.empty else predicted_close

    # Medium-term average (4-10d)
    med_preds = preds[preds["horizon"].between(4, 10)]
    med_avg = med_preds["predicted"].mean() if not med_preds.empty else predicted_close

    # Overall outlook
    outlook_pct = ((preds["predicted"].mean() / ref_price) - 1) * 100 if ref_price else 0
    if outlook_pct > 3:
        outlook = "📈 BULLISH"
    elif outlook_pct > 1:
        outlook = "📈 SLIGHTLY BULLISH"
    elif outlook_pct < -3:
        outlook = "📉 BEARISH"
    elif outlook_pct < -1:
        outlook = "📉 SLIGHTLY BEARISH"
    else:
        outlook = "➡️ RANGEBOUND"

    # Spike overlay info
    overlay_rows = preds[preds["spike_overlay"] > 0]
    overlay_note = ""
    if not overlay_rows.empty:
        max_overlay = overlay_rows["spike_overlay"].max() * 100
        overlay_note = f"\n⚡ Spike overlay active: up to +{max_overlay:.1f}% adjustment"

    # Feature row for weather info
    fr = getattr(predictor, '_forecast_feature_row', None)
    hdd_now = float(fr.get("hdd", 0)) if fr is not None else 0
    hdd_7d = float(fr.get("hdd_7d", 0)) if fr is not None else 0

    # Volatility
    vol_ann = predictor.vol_model.recent_vol * (252 ** 0.5) * 100 if hasattr(predictor, 'vol_model') else 0

    msg = (
        f"🔥 <b>NatGas Forecast Update</b> {dir_emoji}\n"
        f"📅 {_esc(now)}  •  Trigger: {_esc(reason)}\n"
        f"⏱️ Run time: {duration_s:.0f}s\n"
        f"{'━' * 32}\n\n"

        f"💰 <b>Current Price: ${ref_price:.3f}</b>\n"
        f"{dir_emoji} <b>Predicted: ${predicted_close:.3f}  ({change_pct:+.1f}%)</b>\n"
        f"📊 90% CI: ${low5:.3f} – ${high95:.3f}\n"
        f"🎯 Direction: <b>{dir_text}</b>\n\n"

        f"{'━' * 32}\n"
        f"📈 <b>Outlook</b>\n"
        f"  Short (1-3d): ${short_avg:.3f}\n"
        f"  Medium (4-10d): ${med_avg:.3f}\n"
        f"  10d outlook: {outlook} ({outlook_pct:+.1f}%)\n\n"

        f"⚡ <b>Spike Risk: {risk_badge} ({spike_risk:.2f})</b>\n"
        f"  Regime: {_esc(spike_regime)}"
        f"{overlay_note}\n\n"

        f"🌡️ <b>Weather</b>\n"
        f"  HDD today: {hdd_now:.0f}  •  HDD 7d avg: {hdd_7d:.0f}\n"
        f"  Vol (ann): {vol_ann:.0f}%\n"
    )

    # Forecast table (compact)
    msg += f"\n{'━' * 32}\n"
    msg += "📋 <b>Forecast Table</b>\n<pre>"
    msg += f"{'Date':<10} {'Pred':>6} {'Chg%':>6} {'Low':>6} {'High':>6} {'Risk':>4}\n"
    msg += f"{'─'*42}\n"
    for _, row in preds.iterrows():
        d = row["date"][:10] if len(str(row["date"])) >= 10 else str(row["date"])
        p = f"${row['predicted']:.2f}"
        c = f"{row['change_pct']:+.1f}%"
        lo = f"${row['low_5pct']:.2f}"
        hi = f"${row['high_95pct']:.2f}"
        sr = f"{row.get('spike_risk', 0):.2f}"
        msg += f"{d:<10} {p:>6} {c:>6} {lo:>6} {hi:>6} {sr:>4}\n"
    msg += "</pre>"

    return msg


def fmt_error_msg(error: str, context: str = "model run") -> str:
    return (
        f"❌ <b>Error during {_esc(context)}</b>\n\n"
        f"<code>{_esc(error[:500])}</code>\n\n"
        f"⏳ Will retry at next scheduled cycle."
    )


def fmt_help_msg() -> str:
    return (
        "🔥 <b>NatGas Predictor v18 — Commands</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "🟢 <b>Control</b>\n"
        "  /start — Activate model &amp; auto-refresh\n"
        "  /stop — Pause to standby (keeps listening)\n"
        "  /refresh — Force immediate model re-run\n"
        "  /shutdown — Kill the daemon process\n\n"

        "📊 <b>Data &amp; Forecasts</b>\n"
        "  /forecast — Full forecast table (10 horizons)\n"
        "  /intraday — Today's intraday prediction\n"
        "  /summary — Quick one-line snapshot\n\n"

        "⚡ <b>Risk &amp; Analysis</b>\n"
        "  /spike — Spike risk assessment\n"
        "  /weather — Weather forecast &amp; HDD data\n"
        "  /vol — Volatility &amp; risk context\n"
        "  /drivers — Top features driving forecast\n"
        "  /storage — Storage inventory status\n\n"

        "🔧 <b>System</b>\n"
        "  /status — Daemon status, uptime, next run\n"
        "  /backtest YYYY-MM-DD — Run backtest for date\n"
        "  /config — Show current configuration\n"
        "  /help — This message\n\n"

        "📡 Auto-refresh: NWP cycles + EIA storage report\n"
        "🕐 NWP data available at ~03:30, 09:30, 15:30, 21:30 UTC"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_help(runner: "DaemonRunner", args: str):
    runner.bot.send(fmt_help_msg())


def cmd_start(runner: "DaemonRunner", args: str):
    """Activate the model from standby."""
    if runner.state == runner.ACTIVE and not runner._is_running:
        # Already active — just refresh
        runner.bot.send(
            "✅ Already active. Starting a fresh model run...\n"
            "Use /stop first if you want to pause."
        )
        runner._force_refresh = True
        return
    if runner._is_running:
        runner.bot.send("⏳ A model run is already in progress. Please wait.")
        return

    # Activate in a thread so the main loop stays responsive
    def _activate_worker():
        runner.activate()

    t = threading.Thread(target=_activate_worker, name="activation", daemon=True)
    t.start()


def cmd_stop(runner: "DaemonRunner", args: str):
    """Pause to standby — stop auto-refresh but keep listening."""
    if runner.state == runner.STANDBY:
        runner.bot.send("⏸️ Already in standby. Send /start to activate.")
        return
    runner.deactivate()


def cmd_shutdown(runner: "DaemonRunner", args: str):
    """Kill the daemon process entirely."""
    runner.bot.send(
        "🛑 <b>Shutdown requested.</b>\n"
        "Daemon will terminate after this message.\n"
        "Restart the script manually to bring it back."
    )
    runner.stop()


def cmd_status(runner: "DaemonRunner", args: str):
    sched = runner.scheduler
    now_utc = datetime.now(timezone.utc)
    uptime = now_utc - sched.start_time
    uptime_str = f"{uptime.days}d {uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"

    last_run = sched.last_run_utc.strftime("%Y-%m-%d %H:%M UTC") if sched.last_run_utc else "Never"

    # State display
    if runner.state == runner.STANDBY:
        state_icon = "⏸️"
        state_text = "STANDBY (listening only)"
    elif runner._is_running:
        state_icon = "🔄"
        state_text = "ACTIVE — model running..."
    else:
        state_icon = "🟢"
        state_text = "ACTIVE"

    has_preds = runner.last_predictions is not None and not runner.last_predictions.empty

    msg = (
        f"📊 <b>Daemon Status</b>\n"
        f"{'━' * 28}\n"
        f"  {state_icon} State: <b>{state_text}</b>\n"
        f"  ⏱️ Uptime: {_esc(uptime_str)}\n"
        f"  🔁 Total runs: {sched.run_count}\n"
        f"  📅 Last run: {_esc(last_run)}\n"
        f"  📈 Has predictions: {'✅' if has_preds else '❌'}\n"
    )

    if runner.last_run_duration:
        msg += f"  ⏱️ Last run time: {runner.last_run_duration:.0f}s\n"

    # Show next refresh only if active
    if runner.state == runner.ACTIVE:
        next_time, next_reason = sched.next_refresh_utc()
        time_to_next = next_time - now_utc
        mins_left = max(0, int(time_to_next.total_seconds()) // 60)
        msg += (
            f"\n📡 <b>Auto-refresh</b>\n"
            f"  Next: {next_time.strftime('%H:%M UTC')} ({_esc(next_reason)})\n"
            f"  Countdown: ~{mins_left}m\n"
        )
    else:
        msg += "\n⏸️ Auto-refresh paused. Send /start to activate.\n"

    if runner.predictor and hasattr(runner.predictor, 'live_price') and runner.predictor.live_price:
        msg += f"\n💰 Last price: ${runner.predictor.live_price:.3f}"

    runner.bot.send(msg)


def cmd_forecast(runner: "DaemonRunner", args: str):
    preds = runner.last_predictions
    if preds is None or preds.empty:
        runner.bot.send("⚠️ No predictions available yet. Run /refresh to generate.")
        return

    ref = getattr(runner.predictor, '_forecast_current', None) or runner.predictor.live_price
    msg = f"🔮 <b>NatGas Forecast Table</b>\n"
    msg += f"💰 Reference: ${ref:.3f}\n\n" if ref else "\n"
    msg += "<pre>"
    msg += f"{'Date':<12} {'Pred':>7} {'Chg':>7} {'Chg%':>6} {'Low5':>7} {'Hi95':>7} {'Conf':<4}\n"
    msg += f"{'─' * 55}\n"

    for _, r in preds.iterrows():
        d = str(r['date'])[:10]
        wkd = str(r.get('weekday', ''))[:3]
        p = f"${r['predicted']:.3f}"
        chg = f"{r['change']:+.3f}"
        cpct = f"{r['change_pct']:+.1f}%"
        lo = f"${r['low_5pct']:.3f}"
        hi = f"${r['high_95pct']:.3f}"
        conf = str(r.get('confidence', ''))[:4]
        msg += f"{d:<12} {p:>7} {chg:>7} {cpct:>6} {lo:>7} {hi:>7} {conf:<4}\n"

    msg += "</pre>"

    # Summary
    mean_p = preds['predicted'].mean()
    lo_p = preds['predicted'].min()
    hi_p = preds['predicted'].max()
    msg += f"\n📊 Mean: ${mean_p:.3f}  Range: ${lo_p:.3f}–${hi_p:.3f}"

    runner.bot.send(msg)


def cmd_intraday(runner: "DaemonRunner", args: str):
    preds = runner.last_predictions
    if preds is None or preds.empty:
        runner.bot.send("⚠️ No predictions available. Run /refresh first.")
        return

    ref = getattr(runner.predictor, '_forecast_current', None) or runner.predictor.live_price
    first = preds.iloc[0]
    pred = first["predicted"]
    chg = first["change"]
    chg_pct = first["change_pct"]
    lo = first["low_5pct"]
    hi = first["high_95pct"]
    regime = first.get("spike_regime", "NORMAL")
    risk = first.get("spike_risk", 0)
    overlay = first.get("spike_overlay", 0)

    if chg_pct > 1:
        emoji = "🟢"
    elif chg_pct < -1:
        emoji = "🔴"
    else:
        emoji = "⚪"

    vol = runner.predictor.vol_model.recent_vol if hasattr(runner.predictor, 'vol_model') else 0
    sigma_lo = ref * (1 - vol) if ref else 0
    sigma_hi = ref * (1 + vol) if ref else 0

    # Zones
    zone_buy = lo + (pred - lo) * 0.3
    zone_sell = pred + (hi - pred) * 0.7

    msg = (
        f"⚡ <b>Intraday Forecast</b>\n"
        f"{'━' * 30}\n\n"
        f"💰 Open/Ref: <b>${ref:.3f}</b>\n"
        f"{emoji} Target: <b>${pred:.3f}</b>  ({chg_pct:+.2f}%)\n"
        f"📊 Session: ${lo:.3f} – ${hi:.3f}\n"
        f"📐 1σ Range: ${sigma_lo:.3f} – ${sigma_hi:.3f}\n\n"
        f"⚡ Risk: {risk:.2f} ({regime})\n"
    )
    if overlay > 0:
        msg += f"↯ Spike overlay: +{overlay * 100:.1f}%\n"

    msg += (
        f"\n🎯 <b>Zones</b>\n"
        f"  🟢 Buy below: ${zone_buy:.3f}\n"
        f"  ⚪ Neutral: ${zone_buy:.3f}–${zone_sell:.3f}\n"
        f"  🔴 Sell above: ${zone_sell:.3f}\n"
    )

    # Drivers
    drivers = first.get("drivers", [])
    if drivers:
        msg += f"\n📌 <b>Drivers:</b>\n"
        for d in drivers[:4]:
            msg += f"  • {_esc(d)}\n"

    runner.bot.send(msg)


def cmd_summary(runner: "DaemonRunner", args: str):
    preds = runner.last_predictions
    if preds is None or preds.empty:
        runner.bot.send("⚠️ No data. /refresh to generate.")
        return

    ref = getattr(runner.predictor, '_forecast_current', None) or runner.predictor.live_price
    first = preds.iloc[0]
    pred = first["predicted"]
    chg_pct = first["change_pct"]
    risk = first.get("spike_risk", 0)

    mean_pred = preds["predicted"].mean()
    outlook_pct = ((mean_pred / ref) - 1) * 100 if ref else 0

    if outlook_pct > 1:
        arrow = "📈"
    elif outlook_pct < -1:
        arrow = "📉"
    else:
        arrow = "➡️"

    emoji = "🟢" if chg_pct > 1 else ("🔴" if chg_pct < -1 else "⚪")

    msg = (
        f"{emoji} ${ref:.3f} → ${pred:.3f} ({chg_pct:+.1f}%) "
        f"| 10d avg ${mean_pred:.2f} ({outlook_pct:+.1f}%) {arrow} "
        f"| Risk: {risk:.2f}"
    )
    runner.bot.send(msg)


def cmd_spike(runner: "DaemonRunner", args: str):
    preds = runner.last_predictions
    if preds is None or preds.empty:
        runner.bot.send("⚠️ No data.")
        return

    msg = "⚡ <b>Spike Risk Assessment</b>\n"
    msg += f"{'━' * 30}\n\n"

    for _, r in preds.iterrows():
        risk = r.get("spike_risk", 0)
        regime = r.get("spike_regime", "NORMAL")
        overlay = r.get("spike_overlay", 0)
        d = str(r["date"])[:10]
        wkd = str(r.get("weekday", ""))[:3]

        if risk >= 0.7:
            badge = "🔴"
        elif risk >= 0.5:
            badge = "🟠"
        elif risk >= 0.3:
            badge = "🟡"
        else:
            badge = "🟢"

        line = f"{badge} {d} ({wkd}): <b>{risk:.2f}</b> {regime}"
        if overlay > 0:
            line += f"  ↯+{overlay * 100:.1f}%"
        msg += line + "\n"

    # Feature context
    fr = getattr(runner.predictor, '_forecast_feature_row', None)
    if fr is not None:
        pv = fr.get("polar_vortex_flag", 0)
        frz = fr.get("freeze_off_risk", 0)
        stress = fr.get("cold_storage_stress", 0)
        msg += (
            f"\n📌 <b>Preconditions</b>\n"
            f"  Polar vortex: {'🔴 ACTIVE' if pv > 0 else '🟢 inactive'}\n"
            f"  Freeze-off risk: {frz:.2f}\n"
            f"  Cold×storage stress: {stress:.0f}\n"
        )

    runner.bot.send(msg)


def cmd_weather(runner: "DaemonRunner", args: str):
    fr = getattr(runner.predictor, '_forecast_feature_row', None)
    if fr is None:
        runner.bot.send("⚠️ No weather data. Run /refresh first.")
        return

    hdd = fr.get("hdd", 0)
    hdd_7d = fr.get("hdd_7d", 0)
    hdd_14d = fr.get("hdd_14d", 0)
    eff_hdd = fr.get("effective_hdd", hdd)
    temp = fr.get("temp_avg", 0)
    wind = fr.get("wind_avg", 0)
    anom = fr.get("temp_anomaly", 0)
    hdd_delta = fr.get("hdd_delta_7d", 0)
    hdd_30d = fr.get("hdd_30d_total", 0)

    if hdd > 35:
        cold_emoji = "🥶"
    elif hdd > 20:
        cold_emoji = "❄️"
    elif hdd > 5:
        cold_emoji = "🌤️"
    else:
        cold_emoji = "☀️"

    trend = "warming ↗️" if hdd_delta < -3 else ("cooling ↘️" if hdd_delta > 3 else "stable ➡️")

    msg = (
        f"🌡️ <b>Weather Report</b> {cold_emoji}\n"
        f"{'━' * 28}\n\n"
        f"  🌡️ Temp: {temp:.1f}°F\n"
        f"  💨 Wind: {wind:.1f} mph\n"
        f"  🔥 HDD today: <b>{hdd:.1f}</b>\n"
        f"  🔥 Eff HDD (wind-chill): {eff_hdd:.1f}\n"
        f"  📊 HDD 7d avg: {hdd_7d:.1f}\n"
        f"  📊 HDD 14d avg: {hdd_14d:.1f}\n"
        f"  📊 HDD 30d total: {hdd_30d:.0f}\n"
        f"  📈 Trend: {trend} (Δ7d: {hdd_delta:+.1f})\n"
        f"  🌡️ Temp anomaly: {anom:+.1f}°F vs normal\n"
    )

    # Scenario table
    preds = runner.last_predictions
    if preds is not None and not preds.empty:
        ref = getattr(runner.predictor, '_forecast_current', None) or runner.predictor.live_price
        if ref:
            short_avg = preds[preds["horizon"].between(0, 3)]["predicted"].mean()
            msg += (
                f"\n🌡️ <b>Scenarios (3d avg)</b>\n<pre>"
                f"{'Scenario':<25} {'Price':>7} {'vs Ref':>7}\n"
                f"{'─' * 40}\n"
                f"{'10°F colder':<25} ${short_avg * 1.06:>6.3f} {'+6.0%':>7}\n"
                f"{'5°F colder':<25} ${short_avg * 1.03:>6.3f} {'+3.0%':>7}\n"
                f"{'As forecast':<25} ${short_avg:>6.3f} {'base':>7}\n"
                f"{'5°F warmer':<25} ${short_avg * 0.97:>6.3f} {'-3.0%':>7}\n"
                f"{'10°F warmer':<25} ${short_avg * 0.94:>6.3f} {'-6.0%':>7}\n"
                f"</pre>"
            )

    runner.bot.send(msg)


def cmd_vol(runner: "DaemonRunner", args: str):
    if not hasattr(runner.predictor, 'vol_model'):
        runner.bot.send("⚠️ No volatility data.")
        return

    vm = runner.predictor.vol_model
    full_vol = vm.daily_vol * (252 ** 0.5) * 100
    recent_vol = vm.recent_vol * (252 ** 0.5) * 100
    regime = vm.regime.value if hasattr(vm, 'regime') else "unknown"

    msg = (
        f"📊 <b>Volatility &amp; Risk</b>\n"
        f"{'━' * 28}\n\n"
        f"  📈 Full-window vol: {full_vol:.1f}% ann.\n"
        f"  📈 Recent vol (filtered): {recent_vol:.1f}% ann.\n"
        f"  📐 Daily σ: {vm.recent_vol * 100:.2f}%\n"
        f"  🏷️ Regime: <b>{_esc(regime.upper())}</b>\n\n"
        f"  💡 <i>Recent vol filters out spike days (&gt;15% moves)\n"
        f"  for more stable CI estimation.</i>"
    )
    runner.bot.send(msg)


def cmd_drivers(runner: "DaemonRunner", args: str):
    if not hasattr(runner.predictor, 'ensemble') or not runner.predictor.ensemble.models:
        runner.bot.send("⚠️ No model data.")
        return

    fnames = runner.predictor.feat_engine.feature_names
    imp = runner.predictor.ensemble.extract_importance(1)  # 1d horizon
    if imp is None:
        runner.bot.send("⚠️ No feature importance data.")
        return

    pairs = sorted(zip(fnames, imp), key=lambda x: x[1], reverse=True)[:15]
    max_imp = pairs[0][1] if pairs else 1

    msg = "🔬 <b>Top 15 Feature Importances (1d)</b>\n<pre>"
    for name, score in pairs:
        bar_len = int(20 * score / max_imp) if max_imp > 0 else 0
        bar = "█" * bar_len
        msg += f"{name:<25} {bar:<20} {score:.4f}\n"
    msg += "</pre>"

    runner.bot.send(msg)


def cmd_storage(runner: "DaemonRunner", args: str):
    fr = getattr(runner.predictor, '_forecast_feature_row', None)
    if fr is None:
        runner.bot.send("⚠️ No data.")
        return

    bcf = fr.get("storage_bcf", 0)
    vs_avg = fr.get("storage_vs_avg", 0)
    draw = fr.get("storage_draw_rate", 0)
    draw_accel = fr.get("storage_draw_accel", 0)
    stress = fr.get("cold_storage_stress", 0)

    direction = "above" if vs_avg > 0 else "below"
    draw_emoji = "📉" if draw < 0 else "📈"

    msg = (
        f"📦 <b>Storage Report</b>\n"
        f"{'━' * 28}\n\n"
        f"  📦 Inventory: <b>{bcf:.0f} Bcf</b>\n"
        f"  📊 vs 5yr avg: {abs(vs_avg):.0f} Bcf {direction}\n"
        f"  {draw_emoji} Draw rate: {draw:.1f} Bcf/week\n"
        f"  📈 Draw acceleration: {draw_accel:.2f}\n"
        f"  ⚡ Cold×storage stress: {stress:.0f}\n"
    )

    # EIA report timing
    now = datetime.now()
    days_to_thu = (3 - now.weekday()) % 7
    next_report = (now + timedelta(days=days_to_thu)).strftime("%Y-%m-%d")
    if days_to_thu == 0 and now.hour >= 11:
        next_report = (now + timedelta(days=7)).strftime("%Y-%m-%d")
        msg += f"\n📅 Next EIA report: {next_report} (Thu 10:30 ET)\n"
        msg += f"  ✅ Today's report already released"
    elif days_to_thu == 0:
        msg += f"\n📅 EIA report: <b>TODAY</b> at 10:30 ET\n"
        msg += f"  ⚡ Expect 2-8% move at release!"
    else:
        msg += f"\n📅 Next EIA report: {next_report} (Thu 10:30 ET)"

    runner.bot.send(msg)


def cmd_config(runner: "DaemonRunner", args: str):
    cfg = runner.predictor.cfg if runner.predictor else None
    if cfg is None:
        runner.bot.send("⚠️ No configuration loaded.")
        return

    msg = (
        f"⚙️ <b>Configuration</b>\n"
        f"{'━' * 28}\n<pre>"
        f"History:          {cfg.history_years} years\n"
        f"Forecast horizon: {cfg.forecast_horizon} days\n"
        f"Optuna trials:    {cfg.max_optuna_trials}\n"
        f"Weather:          {'ON' if cfg.use_weather else 'OFF'}\n"
        f"Cross-commodity:  {'ON' if cfg.use_cross_commodity else 'OFF'}\n"
        f"Storage:          {'ON' if cfg.use_storage else 'OFF'}\n"
        f"LNG:              {'ON' if cfg.use_lng else 'OFF'}\n"
        f"Live ticker:      {cfg.live_price_ticker}\n"
        f"Weather cities:   {len(cfg.weather_locations)}\n"
        f"</pre>"
    )
    runner.bot.send(msg)


def cmd_refresh(runner: "DaemonRunner", args: str):
    if runner.state == runner.STANDBY:
        runner.bot.send(
            "⏸️ Daemon is in standby. Send /start to activate first,\n"
            "or use /start which includes a fresh run."
        )
        return
    if runner._is_running:
        runner.bot.send("⏳ A model run is already in progress. Your refresh will start when it finishes.")
        runner._force_refresh = True
    else:
        runner.bot.send("🔄 <b>Manual refresh triggered</b>\nStarting full model run...")
        runner._force_refresh = True


def cmd_backtest(runner: "DaemonRunner", args: str):
    date_str = args.strip()
    if not date_str or len(date_str) != 10:
        runner.bot.send("❓ Usage: /backtest YYYY-MM-DD\nExample: /backtest 2026-01-20")
        return

    # Backtest works even from standby — just needs the module loaded
    if runner._predictor_module is None:
        runner.bot.send(f"📦 Loading model module for backtest...")
        if not runner._load_module():
            return

    if runner._is_running:
        runner.bot.send(f"⏳ A run is in progress. Backtest for {_esc(date_str)} queued.")
    else:
        runner.bot.send(f"🔁 <b>Backtest: {_esc(date_str)}</b>\n⏳ Running... (2-5 min)")
    runner._backtest_date = date_str
    # Temporarily go active so the main loop processes the backtest
    if runner.state == runner.STANDBY:
        runner.state = runner.ACTIVE
        runner._return_to_standby_after_backtest = True


# ═══════════════════════════════════════════════════════════════════════════════
#  DAEMON RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class DaemonRunner:
    """
    Main orchestrator with two modes:
      STANDBY — lightweight Telegram listener only (no model loaded, no auto-refresh)
      ACTIVE  — model loaded, auto-refreshes on NWP cycles, full commands available

    Flow:
      daemon starts → STANDBY → user sends /start → first model run → ACTIVE
      /stop → STANDBY (keeps listening, stops auto-refresh)
      /start → ACTIVE again
      /shutdown → kills the process entirely
    """

    # States
    STANDBY = "standby"
    ACTIVE = "active"

    def __init__(self, bot: TelegramBot, run_once: bool = False, auto_start: bool = False):
        self.bot = bot
        self.scheduler = NWPScheduler()
        self.predictor = None
        self.last_predictions = None
        self.last_run_duration: Optional[float] = None
        self._is_running = False
        self._force_refresh = False
        self._backtest_date: Optional[str] = None
        self._shutdown = False  # hard kill
        self._run_once = run_once
        self._auto_start = auto_start  # skip standby, go straight to active
        self._predictor_module = None
        self._return_to_standby_after_backtest = False
        self.state = self.STANDBY

        # Register commands
        commands = {
            "help":      (cmd_help,      "Show available commands"),
            "start":     (cmd_start,     "Activate model & auto-refresh"),
            "stop":      (cmd_stop,      "Pause to standby (keeps listening)"),
            "shutdown":  (cmd_shutdown,   "Kill the daemon process entirely"),
            "status":    (cmd_status,    "Daemon status & uptime"),
            "forecast":  (cmd_forecast,  "Full forecast table"),
            "intraday":  (cmd_intraday,  "Today's intraday prediction"),
            "summary":   (cmd_summary,   "Quick one-line snapshot"),
            "spike":     (cmd_spike,     "Spike risk assessment"),
            "weather":   (cmd_weather,   "Weather forecast & HDD"),
            "vol":       (cmd_vol,       "Volatility & risk context"),
            "drivers":   (cmd_drivers,   "Top feature importances"),
            "storage":   (cmd_storage,   "Storage inventory status"),
            "config":    (cmd_config,    "Current configuration"),
            "refresh":   (cmd_refresh,   "Force immediate refresh"),
            "backtest":  (cmd_backtest,  "Backtest a specific date"),
        }
        for name, (fn, desc) in commands.items():
            bot.register(name, fn, desc)

    def _load_module(self) -> bool:
        """Load the predictor module if not already loaded."""
        if self._predictor_module is not None:
            return True
        try:
            self._predictor_module = _load_predictor_module()
            return True
        except FileNotFoundError as e:
            log.error(str(e))
            self.bot.send(fmt_error_msg(str(e), "module loading"))
            return False

    def _run_predictor(self, backtest_date: str = "") -> bool:
        """Run the full predictor pipeline. Returns True on success."""
        self._is_running = True
        start = time.time()

        try:
            mod = self._predictor_module
            cfg = mod.Config()
            cfg.eia_api_key = os.getenv("EIA_API_KEY", "")
            if backtest_date:
                cfg.backtest_date = backtest_date

            predictor = mod.NatGasPredictor(cfg)

            # Step 1: Fetch all data
            merged = predictor.fetch_all_data()
            if merged.empty:
                raise RuntimeError("Data fetch returned empty")

            # Step 2: Feature engineering
            featured = predictor.build_features()
            if featured.empty:
                raise RuntimeError("Feature engineering failed")

            # Step 3: Train
            if not predictor.train():
                raise RuntimeError("Training failed")

            # Step 4: Backtest (skip in run_once for speed)
            if not self._run_once:
                predictor.backtest(60)

            # Step 5: Forecast
            preds = predictor.forecast()
            if preds.empty:
                raise RuntimeError("Forecast produced no results")

            # Step 6: Generate console report
            predictor.generate_report(preds)

            # Store results
            self.predictor = predictor
            self.last_predictions = preds
            self.last_run_duration = time.time() - start

            return True

        except Exception as e:
            self.last_run_duration = time.time() - start
            log.error(f"Predictor run failed: {e}\n{traceback.format_exc()}")
            return False
        finally:
            self._is_running = False

    def _run_in_background(self, reason: str, backtest_date: str = ""):
        """Run the predictor in a background thread while keeping commands responsive."""

        def _worker():
            self.bot.send(
                f"🔄 <b>Model run starting</b>\n"
                f"📡 Trigger: {_esc(reason)}\n"
                f"⏳ Running full pipeline... (commands still active)",
                silent=(reason not in ("manual refresh", "/start"))
            )
            success = self._run_predictor(backtest_date=backtest_date)
            if success:
                if not backtest_date:
                    self.scheduler.mark_run(reason)
                self.bot.send(fmt_run_complete(
                    self.last_predictions, self.predictor,
                    reason, self.last_run_duration))
            else:
                if not backtest_date:
                    self.scheduler.mark_run(reason)  # Avoid retry spam
                self.bot.send(fmt_error_msg(
                    f"Run failed ({reason})", reason))

            # If this was a backtest triggered from standby, return to standby
            if self._return_to_standby_after_backtest:
                self._return_to_standby_after_backtest = False
                self.state = self.STANDBY
                log.info("Backtest complete → returning to STANDBY")

        t = threading.Thread(target=_worker, name=f"predictor-{reason}", daemon=True)
        t.start()
        return t

    def activate(self):
        """Transition from STANDBY → ACTIVE. Loads module + runs first prediction."""
        if self.state == self.ACTIVE and self.last_predictions is not None:
            self.bot.send("✅ Already active. Use /refresh to force a new run.")
            return

        if not self._load_module():
            return

        self.state = self.ACTIVE
        log.info("State → ACTIVE")

        # First run (synchronous so commands have data immediately after)
        log.info("Starting initial model run...")
        self.bot.send(
            "🚀 <b>Activating NatGas Predictor</b>\n"
            "⏳ Loading data, training model, generating forecast...\n"
            "☕ This takes 2-5 minutes on first run."
        )
        success = self._run_predictor()

        if success:
            self.scheduler.mark_run("/start")
            self.bot.send(fmt_run_complete(
                self.last_predictions, self.predictor,
                "/start activation", self.last_run_duration))
        else:
            self.bot.send(fmt_error_msg("Initial model run failed", "activation"))
            # Stay active so user can /refresh to retry
            self.bot.send("💡 Model is active but failed. Try /refresh to retry.")

    def deactivate(self):
        """Transition from ACTIVE → STANDBY. Stops auto-refresh, keeps listening."""
        self.state = self.STANDBY
        log.info("State → STANDBY")
        self.bot.send(
            "⏸️ <b>Paused to Standby</b>\n\n"
            "Auto-refresh stopped. Telegram commands still active.\n"
            "Last forecast data preserved — /forecast still works.\n"
            "Send /start to reactivate."
        )

    def run(self):
        """Main daemon loop."""
        log.info("=" * 60)
        log.info("NatGas Predictor Daemon starting")
        log.info("=" * 60)

        # Send standby greeting
        self.bot.send(fmt_standby_msg())

        # --run-once or --auto-start: skip standby
        if self._run_once or self._auto_start:
            if not self._load_module():
                return
            self.state = self.ACTIVE
            log.info("Auto-start: running initial model...")
            success = self._run_predictor()
            if success:
                self.scheduler.mark_run("startup")
                self.bot.send(fmt_run_complete(
                    self.last_predictions, self.predictor,
                    "auto-start", self.last_run_duration))
            else:
                self.bot.send(fmt_error_msg("Auto-start model run failed", "startup"))
            if self._run_once:
                log.info("Run-once mode — exiting.")
                return

        # Main loop — always responsive to commands
        log.info("Entering main loop (Telegram listener active)...")
        _active_worker: Optional[threading.Thread] = None

        while not self._shutdown:
            try:
                # ALWAYS check for Telegram commands (both STANDBY and ACTIVE)
                if self.bot.enabled:
                    self.bot.process_commands(self)

                # In STANDBY: just listen for commands, do nothing else
                if self.state == self.STANDBY:
                    time.sleep(3)
                    continue

                # ── ACTIVE MODE below ──────────────────────────────────

                # Don't start a new run if one is already active
                if _active_worker and _active_worker.is_alive():
                    time.sleep(3)
                    continue
                _active_worker = None

                # Check for forced refresh
                if self._force_refresh:
                    self._force_refresh = False
                    _active_worker = self._run_in_background("manual refresh")
                    continue

                # Check for backtest request
                if self._backtest_date:
                    bt_date = self._backtest_date
                    self._backtest_date = None
                    _active_worker = self._run_in_background(f"backtest {bt_date}", bt_date)
                    continue

                # Check NWP scheduler for auto-refresh
                should_run, reason = self.scheduler.should_run_now()
                if should_run:
                    _active_worker = self._run_in_background(reason)
                    continue

                # Idle
                time.sleep(5)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"Main loop error: {e}\n{traceback.format_exc()}")
                time.sleep(30)

        log.info("Daemon shutdown.")
        self.bot.send("🛑 <b>NatGas Predictor Daemon stopped</b>\nProcess terminated. Restart manually.")

    def stop(self):
        """Hard shutdown — kills the process."""
        self._shutdown = True


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NatGas Predictor v18 — Continuous Daemon")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Run without Telegram integration")
    parser.add_argument("--run-once", action="store_true",
                        help="Single run, send notification, then exit")
    parser.add_argument("--auto-start", action="store_true",
                        help="Skip standby — activate model immediately on launch")
    args = parser.parse_args()

    # Load Telegram config
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if args.no_telegram:
        token, chat_id = "", ""

    if not token or not chat_id:
        if not args.no_telegram:
            log.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env. "
                "Running without Telegram. Use --no-telegram to suppress this warning."
            )

    # Check EIA key
    if not os.getenv("EIA_API_KEY"):
        log.error("EIA_API_KEY not set in .env — required for data fetching.")
        sys.exit(1)

    bot = TelegramBot(token, chat_id)

    # --run-once or --no-telegram always implies auto-start
    # (can't send /start from Telegram if it's disabled, and run-once needs to run)
    auto_start = args.auto_start or args.run_once or args.no_telegram

    runner = DaemonRunner(bot, run_once=args.run_once, auto_start=auto_start)

    # Graceful shutdown
    def _signal_handler(sig, frame):
        log.info("Shutdown signal received")
        runner.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    runner.run()


if __name__ == "__main__":
    main()
