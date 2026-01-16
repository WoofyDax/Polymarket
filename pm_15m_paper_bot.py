import os, json, time, math, asyncio, logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp
import websockets
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# =======================
# CONFIG / ENV
# =======================
load_dotenv()

MARKET_SLUG = os.getenv("MARKET_SLUG", "btc-updown-15m-1768523400")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # string ok

# Strategy / risk params (sane defaults)
BASE_SIZE = float(os.getenv("BASE_SIZE", "10"))         # shares per entry
MAX_POS = float(os.getenv("MAX_POS", "200"))           # max shares long
HORIZON_SEC = float(os.getenv("HORIZON_SEC", "10"))    # lookback for spot momentum
ENTRY_EDGE_BPS = float(os.getenv("ENTRY_EDGE_BPS", "45"))  # required edge (bps) above costs
EXIT_EDGE_BPS = float(os.getenv("EXIT_EDGE_BPS", "10"))    # exit when edge shrinks below this (bps)
SAFETY_BPS = float(os.getenv("SAFETY_BPS", "10"))          # extra cushion
IMB_LEVELS = int(os.getenv("IMB_LEVELS", "10"))            # levels for depth imbalance
FLOW_HALFLIFE_SEC = float(os.getenv("FLOW_HALFLIFE_SEC", "4"))  # order-flow decay
MIN_BOOK_QUALITY = float(os.getenv("MIN_BOOK_QUALITY", "0.002")) # min spread to avoid junk? (price units)
TRADING_ON = os.getenv("TRADING_ON", "1") == "1"

# Fee modeling
# If you don't know exact fees, keep this conservative.
TAKER_FEE_BPS_FIXED = float(os.getenv("TAKER_FEE_BPS_FIXED", "150"))

# Coinbase spot feed
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
POLY_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/"
GAMMA_API = "https://gamma-api.polymarket.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# =======================
# Utilities
# =======================
def now_ms() -> int:
    return int(time.time() * 1000)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def bps(x: float) -> float:
    return x * 10_000.0

def decay_weight(dt_sec: float, halflife: float) -> float:
    if halflife <= 0:
        return 0.0
    return 0.5 ** (dt_sec / halflife)


# =======================
# Market State
# =======================
@dataclass
class BookSide:
    # list of (price, size) sorted best->worse
    levels: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class PolyBook:
    bid: BookSide = field(default_factory=BookSide)
    ask: BookSide = field(default_factory=BookSide)
    last_update_ms: int = 0

    def best_bid(self) -> Optional[float]:
        return self.bid.levels[0][0] if self.bid.levels else None

    def best_ask(self) -> Optional[float]:
        return self.ask.levels[0][0] if self.ask.levels else None

    def mid(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return max(0.0, ba - bb)

    def depth_imbalance(self, n: int) -> Optional[float]:
        """(sum bid_size - sum ask_size) / (sum bid_size + sum ask_size) for top n levels"""
        if not self.bid.levels or not self.ask.levels:
            return None
        bid_sz = sum(sz for _, sz in self.bid.levels[:n])
        ask_sz = sum(sz for _, sz in self.ask.levels[:n])
        denom = bid_sz + ask_sz
        if denom <= 0:
            return None
        return (bid_sz - ask_sz) / denom

    def microprice(self, n: int = 1) -> Optional[float]:
        """Microprice using best level sizes: (ask*bid_sz + bid*ask_sz)/(bid_sz+ask_sz)"""
        if len(self.bid.levels) < n or len(self.ask.levels) < n:
            return None
        bid_px, bid_sz = self.bid.levels[0]
        ask_px, ask_sz = self.ask.levels[0]
        denom = bid_sz + ask_sz
        if denom <= 0:
            return None
        return (ask_px * bid_sz + bid_px * ask_sz) / denom


@dataclass
class SpotState:
    price: Optional[float] = None
    returns: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp_sec, logret)


@dataclass
class FlowState:
    """Tracks net order-flow pressure based on size deltas on bid/ask near-touch."""
    # We'll keep last known sizes at each price for bid/ask to compute deltas.
    bid_size_at: Dict[float, float] = field(default_factory=dict)
    ask_size_at: Dict[float, float] = field(default_factory=dict)

    # decayed net pressure accumulator
    flow: float = 0.0
    last_flow_ts: float = field(default_factory=lambda: time.time())

    def update_flow(self, delta: float, ts: float, halflife: float):
        # decay existing flow to now, then add delta
        dt = ts - self.last_flow_ts
        if dt > 0:
            self.flow *= decay_weight(dt, halflife)
        self.flow += delta
        self.last_flow_ts = ts

    def get_flow(self, ts: float, halflife: float) -> float:
        dt = ts - self.last_flow_ts
        if dt > 0:
            return self.flow * decay_weight(dt, halflife)
        return self.flow


@dataclass
class PaperAccount:
    pos: float = 0.0         # shares of YES held
    avg_px: float = 0.0
    cash: float = 0.0
    realized: float = 0.0
    last_trade_ms: int = 0

    def mark_unrealized(self, mid: Optional[float]) -> float:
        if mid is None:
            return 0.0
        return (mid - self.avg_px) * self.pos

    def equity(self, mid: Optional[float]) -> float:
        return self.cash + self.realized + self.mark_unrealized(mid)


# =======================
# Global runtime state
# =======================
poly_book = PolyBook()
spot = SpotState()
flow = FlowState()
acct = PaperAccount()

YES_TOKEN_ID: Optional[str] = None
MARKET_ID: Optional[str] = None
MARKET_TITLE: Optional[str] = None

TRADING_ENABLED = TRADING_ON


# =======================
# Gamma: resolve token IDs from slug
# =======================
async def resolve_market_from_slug(session: aiohttp.ClientSession, slug: str) -> Tuple[str, str, str]:
    """
    Returns (market_id, yes_token_id, title)
    We query Gamma for event/market, then pick the "YES" token/asset id.
    Gamma schema can evolve; this function is defensive.
    """
    # Try common endpoints
    urls = [
        f"{GAMMA_API}/markets?slug={slug}",
        f"{GAMMA_API}/markets?search={slug}",
    ]

    data = None
    for url in urls:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200:
                continue
            j = await resp.json()
            # Gamma sometimes returns list or dict with "markets"
            if isinstance(j, list) and j:
                data = j
                break
            if isinstance(j, dict) and j.get("markets"):
                data = j["markets"]
                break

    if not data:
        raise RuntimeError(f"Could not resolve slug via Gamma: {slug}")

    # Find best match
    m = None
    for cand in data:
        if str(cand.get("slug", "")) == slug:
            m = cand
            break
    if m is None:
        m = data[0]

    market_id = str(m.get("id") or m.get("marketId") or "")
    title = str(m.get("title") or m.get("question") or m.get("name") or slug)

    # Tokens / outcomes can be stored in different shapes:
    # - "tokens": [{"token_id":..., "outcome":"Yes"}, ...]
    # - "outcomes": [{"asset_id":..., "name":"Yes"}, ...]
    # - "clobTokenIds": ["...", "..."] with "outcomes": ["Yes","No"]
    yes_token = None

    tokens = m.get("tokens") or m.get("outcomeTokens") or []
    if isinstance(tokens, list) and tokens:
        for t in tokens:
            out = str(t.get("outcome") or t.get("name") or t.get("label") or "").lower()
            tid = t.get("token_id") or t.get("tokenId") or t.get("asset_id") or t.get("assetId")
            if tid and ("yes" == out or out.startswith("yes")):
                yes_token = str(tid)
                break

    if yes_token is None:
        # Try clobTokenIds + outcomes arrays
        clob_ids = m.get("clobTokenIds") or m.get("clob_token_ids") or []
        outs = m.get("outcomes") or m.get("outcomeNames") or []
        if isinstance(clob_ids, list) and isinstance(outs, list) and len(clob_ids) == len(outs):
            for tid, out in zip(clob_ids, outs):
                if str(out).lower().startswith("yes"):
                    yes_token = str(tid)
                    break

    if yes_token is None:
        raise RuntimeError(f"Could not find YES token id for slug={slug}. Gamma shape changed?")

    return market_id, yes_token, title


# =======================
# Spot BTC feed (Coinbase)
# =======================
async def spot_feed():
    global spot
    async with websockets.connect(COINBASE_WS, ping_interval=20, ping_timeout=20) as ws:
        sub = {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]}
        await ws.send(json.dumps(sub))

        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") != "ticker":
                continue

            px = float(msg["price"])
            t = time.time()

            if spot.price:
                r = math.log(px / spot.price)
                spot.returns.append((t, r))
                # keep only last HORIZON_SEC seconds
                spot.returns = [(ts, rr) for ts, rr in spot.returns if ts > t - HORIZON_SEC]
            spot.price = px


def spot_momentum_bps() -> float:
    """Sum log returns over lookback converted to bps-ish."""
    if not spot.returns:
        return 0.0
    s = sum(r for _, r in spot.returns)
    return s * 10_000.0


# =======================
# Polymarket feed (CLOB WebSocket)
# =======================
def parse_levels(levels) -> List[Tuple[float, float]]:
    out = []
    for lv in levels:
        # lv could be dict {price,size} or list [price,size]
        if isinstance(lv, dict):
            p = float(lv.get("price"))
            s = float(lv.get("size"))
        else:
            p = float(lv[0]); s = float(lv[1])
        out.append((p, s))
    return out


async def polymarket_feed():
    global poly_book, flow

    assert YES_TOKEN_ID is not None, "YES_TOKEN_ID must be resolved before starting feed"

    async with websockets.connect(POLY_WS, ping_interval=20, ping_timeout=20) as ws:
        sub = {"type": "MARKET", "assets_ids": [YES_TOKEN_ID], "auth": None}
        await ws.send(json.dumps(sub))

        while True:
            msg = json.loads(await ws.recv())
            et = msg.get("event_type")

            if et == "book":
                bids = parse_levels(msg.get("bids", []))
                asks = parse_levels(msg.get("asks", []))
                poly_book.bid.levels = bids
                poly_book.ask.levels = asks
                poly_book.last_update_ms = now_ms()

                # refresh flow state reference maps at touch to keep deltas consistent
                # (optional, but helps avoid drift)
                # We'll only store first ~50 levels to bound memory
                flow.bid_size_at = {p: s for p, s in bids[:50]}
                flow.ask_size_at = {p: s for p, s in asks[:50]}

                await maybe_trade()

            elif et == "price_change":
                # price_changes: [{asset_id, price, size, side}, ...]
                pcs = msg.get("price_changes", []) or msg.get("priceChanges", [])
                ts = time.time()
                for ch in pcs:
                    if str(ch.get("asset_id") or ch.get("assetId")) != str(YES_TOKEN_ID):
                        continue
                    price = float(ch.get("price"))
                    size = float(ch.get("size"))
                    side = str(ch.get("side")).upper()

                    # Compute delta vs last known at that price and side
                    if side == "BUY":
                        prev = flow.bid_size_at.get(price, 0.0)
                        delta_sz = size - prev
                        flow.bid_size_at[price] = size
                        # Positive delta on bids = buy pressure
                        flow.update_flow(delta_sz, ts, FLOW_HALFLIFE_SEC)
                    elif side == "SELL":
                        prev = flow.ask_size_at.get(price, 0.0)
                        delta_sz = size - prev
                        flow.ask_size_at[price] = size
                        # Increasing asks is sell pressure => subtract
                        flow.update_flow(-delta_sz, ts, FLOW_HALFLIFE_SEC)

                await maybe_trade()

            elif et == "last_trade_price":
                # Useful if you want to read fee_rate_bps from executions.
                # We keep it simple for now (conservative fixed taker bps).
                await maybe_trade()


# =======================
# Strategy: compute fair, edge, costs, decide
# =======================
def estimate_fair_probability() -> Optional[float]:
    """
    This is NOT "predict BTC".
    It's a lag-catcher fair value proxy:
    - Uses BTC momentum over last HORIZON_SEC
    - Uses Polymarket microstructure (imbalance + flow) to confirm
    Output is a probability in [0,1].
    """
    mid = poly_book.mid()
    if mid is None:
        return None

    mom = spot_momentum_bps()  # can be +/- tens to hundreds in fast moves
    imb = poly_book.depth_imbalance(IMB_LEVELS) or 0.0
    flow_now = flow.get_flow(time.time(), FLOW_HALFLIFE_SEC)

    # Normalize flow: scale by a rough typical size; adjust as you observe real books.
    # This keeps it from dominating.
    flow_scaled = clamp(flow_now / 500.0, -1.0, 1.0)

    # Combine into a fair shift around mid (not around 0.5).
    # We move "fair" toward direction when:
    # - BTC momentum is positive
    # - Polymarket book imbalance supports it
    # - Net flow supports it
    #
    # Coeffs are conservative starter values; you’ll tune from logs.
    shift_bps = 0.35 * mom + 180.0 * imb + 120.0 * flow_scaled  # bps in probability-space
    shift = shift_bps / 10_000.0

    fair = clamp(mid + shift, 0.01, 0.99)
    return fair


def compute_cost_bps() -> Optional[float]:
    sp = poly_book.spread()
    if sp is None:
        return None
    # cost in probability terms: spread + taker fee + safety
    return bps(sp) + TAKER_FEE_BPS_FIXED + SAFETY_BPS


async def paper_buy(price: float, qty: float, chat_send):
    global acct
    # pay taker fee conservatively
    fee = price * qty * (TAKER_FEE_BPS_FIXED / 10_000.0)
    new_pos = acct.pos + qty
    if new_pos <= 0:
        return
    acct.cash -= price * qty
    acct.cash -= fee

    acct.avg_px = (acct.avg_px * acct.pos + price * qty) / new_pos
    acct.pos = new_pos
    acct.last_trade_ms = now_ms()

    await chat_send(f"🟢 PAPER BUY YES {qty:.0f} @ {price:.4f} | fee≈{fee:.4f} | pos={acct.pos:.0f} avg={acct.avg_px:.4f}")


async def paper_sell_all(price: float, chat_send):
    global acct
    if acct.pos <= 0:
        return
    qty = acct.pos
    fee = price * qty * (TAKER_FEE_BPS_FIXED / 10_000.0)

    pnl = (price - acct.avg_px) * qty
    acct.realized += pnl
    acct.cash += price * qty
    acct.cash -= fee

    acct.pos = 0.0
    acct.avg_px = 0.0
    acct.last_trade_ms = now_ms()

    await chat_send(f"🔴 PAPER EXIT ALL @ {price:.4f} | trade_pnl={pnl:.4f} fee≈{fee:.4f} | realized={acct.realized:.4f}")


async def maybe_trade():
    if not TRADING_ENABLED:
        return
    if YES_TOKEN_ID is None:
        return

    bb = poly_book.best_bid()
    ba = poly_book.best_ask()
    mid = poly_book.mid()
    sp = poly_book.spread()

    if bb is None or ba is None or mid is None or sp is None:
        return
    if sp < MIN_BOOK_QUALITY:
        # if spread is microscopic, the book may be weird; skip
        return
    if spot.price is None:
        return

    fair = estimate_fair_probability()
    if fair is None:
        return

    costs = compute_cost_bps()
    if costs is None:
        return

    # Edge measured vs worst execution price (buy at ask, sell at bid)
    entry_edge = bps(fair - ba)  # how underpriced ask is vs fair
    exit_edge = bps(fair - mid)

    # Direction gating via imbalance + flow (order-flow confirmation)
    imb = poly_book.depth_imbalance(IMB_LEVELS) or 0.0
    flow_now = flow.get_flow(time.time(), FLOW_HALFLIFE_SEC)

    # Basic confirmation: don't buy unless book/flow not strongly bearish
    confirm_ok = (imb > -0.10) and (flow_now > -50)

    # ENTRY: buy YES if edge exceeds costs+threshold and confirmation ok
    if entry_edge > (costs + ENTRY_EDGE_BPS) and confirm_ok and acct.pos < MAX_POS:
        qty = min(BASE_SIZE, MAX_POS - acct.pos)
        await paper_buy(ba, qty, telegram_broadcast)
        return

    # EXIT: if we’re long and edge shrinks or turns against us, exit at bid (conservative)
    if acct.pos > 0:
        # If fair is no longer above mid by EXIT_EDGE_BPS, take it off.
        if exit_edge < EXIT_EDGE_BPS:
            await paper_sell_all(bb, telegram_broadcast)
            return


# =======================
# Telegram interface
# =======================
telegram_app: Optional[Application] = None

async def telegram_broadcast(text: str):
    global TELEGRAM_CHAT_ID, telegram_app
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        await telegram_app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        logging.warning(f"Telegram send failed: {e}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bb = poly_book.best_bid()
    ba = poly_book.best_ask()
    mid = poly_book.mid()
    sp = poly_book.spread()
    fair = estimate_fair_probability()
    costs = compute_cost_bps() or 0.0

    mom = spot_momentum_bps()
    imb = poly_book.depth_imbalance(IMB_LEVELS) or 0.0
    fl = flow.get_flow(time.time(), FLOW_HALFLIFE_SEC)

    lines = []
    lines.append(f"Market: {MARKET_TITLE or MARKET_SLUG}")
    lines.append(f"YES token: {YES_TOKEN_ID}")
    lines.append(f"Spot BTC: {spot.price}")
    lines.append(f"Book: bid={bb} ask={ba} mid={mid} spread={sp} (bps={bps(sp) if sp else None})")
    lines.append(f"Signal: mom={mom:.1f}bps imb={imb:+.3f} flow={fl:+.1f}")
    lines.append(f"Fair: {fair} | costs≈{costs:.1f}bps")
    lines.append(f"Paper pos: {acct.pos:.0f} avg={acct.avg_px:.4f} realized={acct.realized:.4f} equity={acct.equity(mid):.4f}")
    lines.append(f"Trading enabled: {TRADING_ENABLED}")
    await update.message.reply_text("\n".join(lines))


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mid = poly_book.mid()
    unreal = acct.mark_unrealized(mid)
    eq = acct.equity(mid)
    await update.message.reply_text(
        f"realized={acct.realized:.4f}\n"
        f"unrealized={unreal:.4f}\n"
        f"cash={acct.cash:.4f}\n"
        f"equity={eq:.4f}\n"
        f"pos={acct.pos:.0f} avg={acct.avg_px:.4f}"
    )


async def cmd_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADING_ENABLED
    TRADING_ENABLED = True
    await update.message.reply_text("✅ Trading enabled (paper).")


async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TRADING_ENABLED
    TRADING_ENABLED = False
    await update.message.reply_text("⛔ Trading disabled.")


# =======================
# Main runner
# =======================
async def main():
    global YES_TOKEN_ID, MARKET_ID, MARKET_TITLE, telegram_app

    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        return
    if not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_CHAT_ID not set")
        return

    async with aiohttp.ClientSession() as session:
        market_id, yes_token, title = await resolve_market_from_slug(session, MARKET_SLUG)
        MARKET_ID = market_id
        YES_TOKEN_ID = yes_token
        MARKET_TITLE = title

    # Telegram bot setup
    telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    telegram_app.add_handler(CommandHandler("status", cmd_status))
    telegram_app.add_handler(CommandHandler("pnl", cmd_pnl))
    telegram_app.add_handler(CommandHandler("on", cmd_on))
    telegram_app.add_handler(CommandHandler("off", cmd_off))

    await telegram_broadcast(f"🚀 Started PM 15m PAPER bot\nSlug: {MARKET_SLUG}\nMarket: {MARKET_TITLE}\nYES token: {YES_TOKEN_ID}")

    # Run everything concurrently:
    # - Telegram polling
    # - Spot feed
    # - Polymarket feed
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling()

    try:
        await asyncio.gather(
            spot_feed(),
            polymarket_feed(),
        )
    finally:
        await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
