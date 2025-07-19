from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, PositionIntent
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.trading.requests import GetOrderByIdRequest
from alpaca.data.timeframe import TimeFrame
import asyncio
import numpy as np
from alpaca.common.exceptions import APIError
from datetime import datetime, timedelta
from typing import Dict


def adjust_price_limits(
    predict_price_low,
    predict_price,
    predict_price_high,
    entry_price,
    direction="long",
    take_profit_pct=0.03,
    stop_loss_pct=0.03,
):
    DiffVWAPPred = predict_price - entry_price
    if not predict_price_low < predict_price < predict_price_high:
        if direction == "long":
            predict_price_high = DiffVWAPPred * 1.2 + entry_price
            predict_price_low = DiffVWAPPred * 0.8 + entry_price
        elif direction == "short":
            predict_price_low = DiffVWAPPred * 1.2 + entry_price
            predict_price_high = DiffVWAPPred * 0.8 + entry_price

    predict_price_low = np.clip(
        predict_price_low, predict_price * (1 - stop_loss_pct), None
    )
    predict_price_high = np.clip(
        predict_price_high, None, predict_price * (1 + take_profit_pct)
    )
    entry_price = entry_price + 0.01 if direction == "long" else entry_price - 0.01
    return predict_price_low, predict_price_high, entry_price


def calculate_target_prices(
    prediction, current_price, open_price, take_profit_weight, stop_loss_pct
):
    """Calculate target prices for take profit and stop loss based on enhanced strategy"""

    # Take profit: Open + weight * DiffVWAPPred (from backtest optimization)
    take_profit_price = round(open_price + take_profit_weight * prediction, 2)

    # Stop loss: percentage below entry price
    stop_loss_price = round(current_price * (1 - stop_loss_pct / 100), 2)

    return take_profit_price, stop_loss_price


def determine_equity_allocation(
    trading_client,
    max_positions=1,
    notional=None,
    percentage_notional=100,
    weights=None,
    no_margin=False,
):
    if weights is None:
        weights = np.ones(max_positions) / max_positions
    else:
        weights = np.array(weights) / np.sum(weights)
    if notional is None:
        account = trading_client.get_account()
        if no_margin:
            notional = (
                float(account.non_marginable_buying_power) * percentage_notional / 100
            )
        else:
            notional = (
                float(account.daytrading_buying_power) * percentage_notional / 100
            )

    equities = notional * weights
    return equities


def check_orders(trading_client):
    orders = trading_client.get_orders()
    return orders


def cancel_orders(trading_client):
    trading_client.cancel_orders()


def close_all(trading_client):
    trading_client.close_all_positions(cancel_orders=True)


def ask_price(historical_client, tickers, target_time=None):
    """
    Get current price or opening price for given tickers.

    Args:
        historical_client: Alpaca historical data client
        tickers: List of ticker symbols or single ticker string
        target_time: datetime object. If None, get latest trade price.
                    If provided, get opening price for that date.

    Returns:
        Dict mapping tickers to prices
    """
    if target_time is None:
        # Get latest trade prices (existing logic)
        req = StockLatestTradeRequest(symbol_or_symbols=tickers)
        trades = historical_client.get_stock_latest_trade(req)
        return {t: trades[t].price for t in tickers}
    else:
        # Get opening prices for the specified date
        if isinstance(tickers, str):
            tickers = [tickers]

        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=target_time.date().strftime("%Y-%m-%d"),
            end=(target_time + timedelta(days=1)).date().strftime("%Y-%m-%d"),
        )

        try:
            bars = historical_client.get_stock_bars(req).df.reset_index()
            prices = {k: v for k, v in zip(bars["symbol"], bars["open"])}
            return prices

        except Exception as e:
            print(f"Error getting opening prices for {target_time.date()}: {e}")
            # Fallback to latest trade prices
            req = StockLatestTradeRequest(symbol_or_symbols=tickers)
            trades = historical_client.get_stock_latest_trade(req)
            return {t: trades[t].price for t in tickers}


async def wait_for_order(order_id, trading_client, timeout=60):
    """
    Wait for a specific order to be filled.

    Args:
        order_id (str): The ID of the order to wait for.
        trading_client (TradingClient): The Alpaca trading client instance.
    """
    time_start = datetime.now()
    while True:
        order = trading_client.get_order_by_id(order_id)
        if order.status == "filled":
            print(f"Order {order_id} ({order.symbol} {order.side} {order.qty}) filled.")
            return order  # Return the filled order object
        elif order.status in ["canceled", "expired", "rejected"]:
            print(
                f"Order {order_id} ({order.symbol} {order.side}) did not fill ({order.status})."
            )
            return None  # Indicate order did not fill successfully
        elif order.status == "partially_filled":
            if datetime.now() > time_start + timedelta(minutes=timeout):
                # cancel the order
                trading_client.cancel_order_by_id(order_id)
                print(
                    f"Order {order_id} ({order.symbol} {order.side}) partially filled ({order.status}), cancel the remaining order."
                )
                return order  # Return the partially filled order object
        await asyncio.sleep(1)  # Check status every second


async def async_entry_single(
    trading_client,
    symbol,
    notional,
    limit_price,
    side,
    extended_hours=False,
    debug=False,
):
    """
    Unified entry function for both long buy and short sell.
    """
    # Debug price adjustment
    if debug:
        limit_price = (
            round(limit_price * 10, 2)
            if side == OrderSide.BUY
            else round(limit_price / 10, 2)
        )
    # Calculate quantity
    qty = int(notional / limit_price)
    if qty == 0:
        qty = 1
        print(f"Adjusted qty for {symbol} to 1 due to insufficient notional.")
    # Build limit order request
    if limit_price == "market":
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            extended_hours=extended_hours,
        )
    else:
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            extended_hours=extended_hours,
            limit_price=round(limit_price, 2),
        )
    try:
        print(
            f"Submitting {side.value} entry order for {symbol}: {qty} at limit {limit_price}"
        )
        submitted = trading_client.submit_order(order_request)
        filled = await wait_for_order(submitted.id, trading_client)
        return filled
    except APIError as e:
        print(f"API Error on entry {side.value} for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error on entry {side.value} for {symbol}: {e}")
        return None


async def async_exit_single(
    trading_client,
    symbol,
    qty,
    price,
    side,
    replace_order_id=None,
    extended_hours=False,
    debug=False,
):
    """
    Unified exit function for both sell and buy to cover.
    """
    action = "sell" if side == OrderSide.SELL else "buy to cover"
    time_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{time_start}: Attempting to {action} {qty} of {symbol} at {price} (replacing {replace_order_id or 'N/A'})"
    )
    # Pre-cancellation
    if replace_order_id:
        try:
            trading_client.cancel_order_by_id(replace_order_id)
            print(f"Cancelled previous order {replace_order_id} for {symbol}.")
            await asyncio.sleep(0.5)

        except APIError:
            print(f"Warning: Position {symbol} seems closed. Aborting {action}.")
            return None
    # Determine order type
    is_market = isinstance(price, str) and price.lower() == "market"
    if is_market:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.CLS,
        )
        price_str = "market"
    else:
        # Debug price adjustment
        if debug:
            price = (
                round(price * 10, 2) if side == OrderSide.BUY else round(price / 10, 2)
            )
        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(price, 2),
            extended_hours=extended_hours,
        )
        price_str = f"limit {price}"
    try:
        print(f"Placing new {action} order for {qty} of {symbol} at {price_str}")
        order = trading_client.submit_order(order_request)
        return order.id
    except APIError as e:
        print(f"API Error {action} {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error {action} {symbol}: {e}")
        return None


async def trade_single_stock_shortlong(
    trading_client,
    symbol,
    equity_per_position,
    entry_limit_price,
    exit_limit_price_high,
    exit_limit_price_low,
    direction="long",
    n_periods=15,
    extended_hours=False,
    debug=False,
    historical_client=None,
):
    """
    Function to trade a single stock with support for both long and short directions.

    For long trades: Buy first, then place decreasing limit sell orders over time.
    For short trades: Short sell first, then place increasing limit buy orders over time.

    Args:
        trading_client (TradingClient): The Alpaca trading client instance.
        symbol (str): Stock symbol.
        equity_per_position (float): Amount to invest/short in dollars.
        entry_limit_price (float): Limit price for entry (buy for long, sell for short).
        exit_limit_price_high (float): Highest exit price target (sell for long, buy for short).
        exit_limit_price_low (float): Lowest exit price target (sell for long, buy for short).
        direction (str): Trade direction - "long" or "short". Default is "long".
        n_periods (int): Number of price steps between high and low exit prices.
        extended_hours (bool): Whether to trade in extended hours.
        debug (bool): If True, adjust prices for debugging.
        historical_client: Client for fetching real-time market data.
    """
    time_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_long = direction.lower() == "long"
    side = OrderSide.BUY if is_long else OrderSide.SELL
    exit_side = OrderSide.SELL if is_long else OrderSide.BUY

    # Parameter labels based on direction for better logging
    entry_type = "buy" if is_long else "short"
    exit_type = "sell" if is_long else "buy to cover"

    print(
        f"{time_start}: Start trading {symbol} {side.value} with {entry_type}_limit_price={entry_limit_price}, "
        f"{exit_type}_limit_price_high={exit_limit_price_high} and {exit_type}_limit_price_low={exit_limit_price_low}"
    )

    # Step 1: Execute entry order (buy for long, short sell for short)
    filled_entry_order = await async_entry_single(
        trading_client=trading_client,
        symbol=symbol,
        notional=equity_per_position,
        limit_price=entry_limit_price,
        side=side,
        extended_hours=extended_hours,
        debug=debug,
    )

    if filled_entry_order is None:
        print(
            f"{entry_type.capitalize()} order for {symbol} did not fill. Aborting trade."
        )
        return

    # Step 2: Calculate exit prices and timing
    # For long: prices decrease over time (exit_limit_price_high to exit_limit_price_low)
    # For short: prices increase over time (exit_limit_price_low to exit_limit_price_high)
    if is_long:
        exit_prices = np.linspace(
            exit_limit_price_high, exit_limit_price_low, n_periods
        )
    else:
        exit_prices = np.linspace(
            exit_limit_price_low, exit_limit_price_high, n_periods
        )

    qty = abs(float(filled_entry_order.qty))
    exit_order_id = None
    now_time = datetime.now()
    market_close_time = now_time.replace(hour=16, minute=0, second=0, microsecond=0)
    time_until_close = market_close_time - now_time
    if not debug:
        assert time_until_close.total_seconds() > 0, "Market is already closed"
    # Calculate time period between price adjustments
    period = max(
        1, time_until_close.total_seconds() // n_periods
    )  # Ensure period is at least 1 second

    # Loop through exit periods
    for i, exit_price in enumerate(exit_prices):
        # Check if previous exit order filled
        if exit_order_id:
            try:
                order = trading_client.get_order_by_id(exit_order_id)
                if order.status == "filled":
                    print(
                        f"{exit_type.capitalize()} order {exit_order_id} for {symbol} filled at {order.filled_avg_price}. Trade complete."
                    )
                    break  # Exit loop if filled
                elif order.status in ["canceled", "expired", "rejected"]:
                    print(
                        f"Previous {exit_type} order {exit_order_id} for {symbol} is {order.status}. Placing new order."
                    )
                    exit_order_id = None  # Reset exit_order_id to place a new one
            except APIError as e:
                print(
                    f"Error checking {exit_type} order {exit_order_id} status for {symbol}: {e}. Assuming not filled."
                )
                continue

        # Determine exit price for this period
        if i == n_periods - 1:  # Last period, use market order with CLS
            print(
                f"Last period for {symbol}, using market order with CLS time in force."
            )
            exit_order_id = await async_exit_single(
                trading_client=trading_client,
                symbol=symbol,
                qty=qty,
                price="market",
                side=exit_side,
                replace_order_id=exit_order_id,
                extended_hours=extended_hours,
                debug=debug,
            )
            if exit_order_id is None:
                print(
                    f"Failed to place market {exit_type} order for {symbol}. Trade complete."
                )
            break  # Exit loop after placing market order
        else:
            exit_price = round(exit_price, 2)

        if (exit_limit_price_low == exit_limit_price_high) and (i > 0):
            print(
                f"Waiting {period} seconds before next {exit_type} price update for {symbol}..."
            )
            await asyncio.sleep(period)
            continue
        # Place or replace the exit order
        try:
            exit_order_id = await async_exit_single(
                trading_client=trading_client,
                symbol=symbol,
                qty=qty,
                price=exit_price,
                side=exit_side,
                replace_order_id=exit_order_id,
                extended_hours=extended_hours,
                debug=debug,
            )
        except Exception as e:
            print(f"Error placing {exit_type} order for {symbol}: {e}")
            print(
                f"Waiting {period} seconds before next {exit_type} price update for {symbol}..."
            )
            await asyncio.sleep(period)
            continue

        if exit_order_id is None:
            print(
                f"Failed to place/replace {exit_type} order for {symbol} at price {exit_price}. Trade complete."
            )
            break  # Exit loop after fallback attempt

        # Wait for the period duration, unless it's the last period
        if i < n_periods - 1:
            print(
                f"Waiting {period} seconds before next {exit_type} price update for {symbol}..."
            )
            await asyncio.sleep(period)

    # After loop finishes (or breaks), ensure any final pending order is handled or position is closed
    print(f"{exit_type.capitalize()} loop finished for {symbol}.")


async def trade_single_stock_pred_hl_shortlong(
    trading_client,
    symbol,
    notional,
    entry_limit_price,  # Entry order limit price (buy for long, sell for short)
    take_profit_price,  # Price at which to take profit (close position)
    stop_loss_price,  # Price at which to stop loss (close position)
    direction="long",  # Trade direction: 'long' or 'short'
    extended_hours=False,
    debug=False,
):
    """
    Trades a single stock with an OCO bracket order supporting both long and short directions.

    Args:
        trading_client (TradingClient): The Alpaca trading client instance.
        symbol (str): Stock symbol.
        notional (float): Dollar amount to trade.
        entry_limit_price (float): Entry order limit price (buy for long, sell for short).
        take_profit_price (float): Price for taking profit (closing position).
        stop_loss_price (float): Price for stop loss (closing position).
        direction (str): Trade direction, 'long' or 'short'. Default 'long'.
        extended_hours (bool): Whether trades are allowed in extended hours.
        debug (bool): If True, adjust prices for debugging.
    Returns:
        The submitted bracket order object or None.
    """
    time_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_long = direction.lower() == "long"
    entry_type = "buy" if is_long else "short"
    print(
        f"{time_start}: Placing bracket {entry_type} order for {symbol} with notional ${notional:.2f}, "
        f"entry_limit={entry_limit_price:.2f}, take_profit={take_profit_price:.2f}, stop_loss={stop_loss_price:.2f}"
    )

    # Adjust prices for debug if needed
    if not debug:
        limit_price = entry_limit_price
    else:
        limit_price = (
            round(entry_limit_price * 10, 2)
            if is_long
            else round(entry_limit_price / 10, 2)
        )

    # Verify bracket price levels
    if is_long:
        valid = take_profit_price > entry_limit_price > stop_loss_price
    else:
        valid = take_profit_price < entry_limit_price < stop_loss_price
    if not valid:
        print(
            f"Invalid bracket prices for {symbol} in {direction} direction: TP {take_profit_price}, Entry {entry_limit_price}, SL {stop_loss_price}. Skipping."
        )
        return None

    qty = int(round(notional / limit_price))
    if qty == 0:
        qty = 1
        print(f"Adjusted qty for {symbol} to 1 due to insufficient notional.")
    # Create a bracket order (primary buy/sell + OCO TP/SL)
    bracket_order = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if is_long else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        order_class="bracket",
        limit_price=round(limit_price, 2),
        take_profit=TakeProfitRequest(
            limit_price=round(take_profit_price, 2), time_in_force=TimeInForce.GTC
        ),
        stop_loss=StopLossRequest(
            stop_price=round(stop_loss_price, 2),
            limit_price=None,  # Using market order for stop loss
            time_in_force=TimeInForce.GTC,
        ),
        extended_hours=extended_hours,
    )
    try:
        order = trading_client.submit_order(bracket_order)
        print(f"Bracket order submitted for {symbol}: {order.id}")
        return order
    except APIError as e:
        print(f"API Error submitting bracket order for {symbol}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error placing bracket order for {symbol}: {str(e)}")
        return None


class OrderTracker:
    """Order tracking class for managing bracket orders and EOD exit with lazy loading of child orders"""

    def __init__(
        self,
        trading_client,
        exit_hour: int = 14,
        exit_minute: int = 59,
        debug: bool = False,
    ):
        self.trading_client = trading_client
        self.tracked_orders: Dict[str, Dict] = {}  # symbol -> order_info
        self.exit_hour = exit_hour
        self.exit_minute = exit_minute
        self.debug = debug
        self.eod_task = None

    async def _start_eod_monitoring(self):
        """Start the EOD monitoring task"""
        try:
            self.eod_task = asyncio.create_task(
                self.eod_force_exit_monitor(
                    self.exit_hour, self.exit_minute, self.debug
                )
            )
            print(
                f"[ORDER TRACKER] EOD monitoring started for {self.exit_hour:02d}:{self.exit_minute:02d}"
            )
            await self.eod_task
        except Exception as e:
            print(f"[ORDER TRACKER] Error starting EOD monitoring: {str(e)}")

    async def stop_eod_monitoring(self):
        """Stop the EOD monitoring task"""
        if self.eod_task and not self.eod_task.done():
            self.eod_task.cancel()
            try:
                await self.eod_task
            except asyncio.CancelledError:
                print("[ORDER TRACKER] EOD monitoring task cancelled")

    async def track_bracket_order(
        self, symbol: str, main_order_id: str, direction: str, shares: int
    ):
        """
        Track a bracket order - stores main order info and will fetch child order IDs when needed
        """

        # Store basic tracking information without fetching child orders immediately
        self.tracked_orders[symbol] = {
            "main_order_id": main_order_id,
            "take_profit_id": None,
            "stop_loss_id": None,
            "symbol": symbol,
            "direction": direction,
            "shares": shares,
            "submitted_time": datetime.now(),
            "status": "active",
            "child_orders_fetched": False,  # Flag to track if we've tried to fetch child orders
        }

        print(f"[ORDER TRACKER] Tracking {symbol}:")
        print(f"  Main Order: {main_order_id}")
        print("  Child orders will be fetched when needed")

        return True

    async def _fetch_child_orders(self, symbol: str) -> bool:
        """
        Private method to fetch child order IDs for a bracket order
        Returns True if successful, False otherwise
        """
        if symbol not in self.tracked_orders:
            return False

        order_info = self.tracked_orders[symbol]

        # Skip if already fetched
        if order_info.get("child_orders_fetched", False):
            return True

        try:
            # Get the main order with nested=True to retrieve child orders
            order_request = GetOrderByIdRequest(nested=True)
            main_order = self.trading_client.get_order_by_id(
                order_info["main_order_id"], filter=order_request
            )

            # Extract child order IDs from legs
            take_profit_id = None
            stop_loss_id = None

            if hasattr(main_order, "legs") and main_order.legs:
                for leg in main_order.legs:
                    # Determine order type based on side and direction
                    if leg.side.value.upper() != main_order.side.value.upper():
                        # This is an exit order (opposite side of main order)
                        if hasattr(leg, "order_type") and leg.order_type:
                            if "limit" in leg.order_type.value.lower():
                                take_profit_id = leg.id
                            elif "stop" in leg.order_type.value.lower():
                                stop_loss_id = leg.id
                        else:
                            # Fallback: determine by price comparison if possible
                            if hasattr(leg, "limit_price") and leg.limit_price:
                                take_profit_id = leg.id
                            elif hasattr(leg, "stop_price") and leg.stop_price:
                                stop_loss_id = leg.id

            # Update tracking information with child order IDs
            self.tracked_orders[symbol].update(
                {
                    "take_profit_id": take_profit_id,
                    "stop_loss_id": stop_loss_id,
                    "child_orders_fetched": True,
                }
            )

            print(f"[ORDER TRACKER] Fetched child orders for {symbol}:")
            print(f"  Take Profit: {take_profit_id}")
            print(f"  Stop Loss: {stop_loss_id}")

            return True

        except Exception as e:
            print(
                f"[ORDER TRACKER] Failed to fetch child orders for {symbol}: {str(e)}"
            )
            # Mark as attempted to avoid repeated failures
            self.tracked_orders[symbol]["child_orders_fetched"] = True
            return False

    def get_active_orders(self) -> Dict[str, Dict]:
        """Get currently active tracked orders"""
        return {
            symbol: info
            for symbol, info in self.tracked_orders.items()
            if info["status"] == "active"
        }

    def mark_order_completed(self, symbol: str):
        """Mark an order as completed (filled or cancelled)"""
        if symbol in self.tracked_orders:
            self.tracked_orders[symbol]["status"] = "completed"
            print(f"[ORDER TRACKER] Marked {symbol} as completed")

    async def check_order_status(self, symbol: str) -> str:
        """Check the current status of an order and fetch child orders if needed"""
        if symbol not in self.tracked_orders:
            return "unknown"

        order_info = self.tracked_orders[symbol]
        try:
            # Check main order status first
            main_order = self.trading_client.get_order_by_id(
                order_info["main_order_id"]
            )

            # If main order is filled, try to fetch child orders if not already done
            if main_order.status == "filled" and not order_info.get(
                "child_orders_fetched", False
            ):
                await self._fetch_child_orders(symbol)

            # If main order is completed, mark as completed
            if main_order.status in ["filled", "cancelled", "expired", "rejected"]:
                self.mark_order_completed(symbol)
                return main_order.status

            return main_order.status

        except Exception as e:
            print(f"[ORDER TRACKER] Error checking status for {symbol}: {str(e)}")
            return "error"

    async def cancel_all_orders_for_symbol(self, symbol: str) -> bool:
        """Cancel all orders (main, take profit, stop loss) for a symbol"""
        if symbol not in self.tracked_orders:
            print(f"[ORDER TRACKER] No tracked orders for {symbol}")
            return False

        order_info = self.tracked_orders[symbol]
        cancelled_any = False

        # Try to fetch child orders if not already fetched
        if not order_info.get("child_orders_fetched", False):
            await self._fetch_child_orders(symbol)

        # Collect all order IDs to cancel
        order_ids_to_cancel = []

        if order_info["take_profit_id"]:
            order_ids_to_cancel.append(order_info["take_profit_id"])
        if order_info["stop_loss_id"]:
            order_ids_to_cancel.append(order_info["stop_loss_id"])

        # Cancel each order
        for order_id in order_ids_to_cancel:
            try:
                self.trading_client.cancel_order_by_id(order_id)
                print(f"[ORDER TRACKER] Cancelled order {order_id} for {symbol}")
                cancelled_any = True
            except Exception as e:
                # Don't print error for orders that are already filled/cancelled

                print(
                    f"[ORDER TRACKER] Error cancelling order {order_id} for {symbol}: {str(e)}"
                )

        return cancelled_any

    async def eod_force_exit_monitor(
        self, exit_hour: int, exit_minute: int, debug: bool = False
    ):
        """
        Monitor for EOD time and force exit any remaining positions
        """
        # Calculate target exit time
        now = datetime.now()
        exit_time = now.replace(
            hour=exit_hour, minute=exit_minute, second=0, microsecond=0
        )

        # If exit time has already passed today, don't wait
        if (exit_time <= now) and not debug:
            print(
                f"[EOD MONITOR] Exit time {exit_time.strftime('%H:%M')} has already passed"
            )
            return

        # Calculate wait time
        wait_seconds = (exit_time - now).total_seconds()

        print(f"[EOD MONITOR] Will force exit at {exit_time.strftime('%H:%M:%S')}")
        print(f"[EOD MONITOR] Waiting {wait_seconds:.0f} seconds until forced exit")

        if debug:
            print("[EOD MONITOR] Debug mode: reducing wait time to 10 seconds")
            wait_seconds = min(wait_seconds, 10)

        # Wait until exit time
        await asyncio.sleep(wait_seconds)

        # Force exit all remaining positions
        await self.force_exit_all_positions()

    async def force_exit_all_positions(self):
        """
        Cancel all pending orders and submit market sell orders for remaining positions
        """
        print(
            f"[EOD FORCE EXIT] Starting forced exit at {datetime.now().strftime('%H:%M:%S')}"
        )

        active_orders = self.get_active_orders()

        if not active_orders:
            print("[EOD FORCE EXIT] No active orders to process")
            return

        for symbol, order_info in active_orders.items():
            try:
                print(f"[EOD FORCE EXIT] Processing {symbol}...")

                # First, check if the order is still active
                order_status = await self.check_order_status(symbol)

                if order_status in ["filled"]:
                    print(f"[EOD FORCE EXIT] {symbol}: Order already {order_status}")

                    # Cancel all related orders (main, take profit, stop loss)
                    cancelled = await self.cancel_all_orders_for_symbol(symbol)

                    if cancelled:
                        print(
                            f"[EOD FORCE EXIT] Successfully cancelled orders for {symbol}"
                        )

                        # Submit market order to close position if we have open positions
                        try:
                            # Determine order side to close position based on original direction
                            if order_info["direction"] == "long":
                                # Long position - sell to close
                                close_side = OrderSide.SELL
                                position_intent = PositionIntent.SELL_TO_CLOSE
                            else:
                                # Short position - buy to cover
                                close_side = OrderSide.BUY
                                position_intent = PositionIntent.BUY_TO_CLOSE

                            market_close_order = MarketOrderRequest(
                                symbol=symbol,
                                qty=order_info["shares"],
                                side=close_side,
                                time_in_force=TimeInForce.DAY,
                                position_intent=position_intent,
                            )

                            close_order = self.trading_client.submit_order(
                                order_data=market_close_order
                            )
                            print(
                                f"[EOD FORCE EXIT] Market close order submitted for {symbol}: {close_order.id}"
                            )

                        except Exception as e:
                            print(
                                f"[EOD FORCE EXIT] Error submitting market close order for {symbol}: {str(e)}"
                            )
                    else:
                        print(f"[EOD FORCE EXIT] Failed to cancel orders for {symbol}")
                else:
                    print(f"[EOD FORCE EXIT] {symbol}: Order is {order_status}")

            except Exception as e:
                print(f"[EOD FORCE EXIT] Error processing {symbol}: {str(e)}")

        print(
            f"[EOD FORCE EXIT] Forced exit completed at {datetime.now().strftime('%H:%M:%S')}"
        )


async def place_native_bracket_order(
    trading_client,
    symbol,
    equity,
    entry_price,
    take_profit_price,
    stop_loss_price,
    direction,
    order_tracker: OrderTracker,
    extended_hours=False,
):
    """
    Place native Alpaca bracket order and track it with OrderTracker
    """

    # Calculate position size
    shares = int(equity / entry_price * 0.95)
    if shares <= 0:
        print(f"[ERROR] Invalid share count for {symbol}: {shares}")
        return None

    print(f"[NATIVE BRACKET ORDER] {symbol}: {shares} shares @ ${entry_price:.2f}")
    print(f"  Take Profit: ${take_profit_price:.2f}")
    print(f"  Stop Loss: ${stop_loss_price:.2f}")

    # Adjust order parameters based on direction
    if direction == "long":
        order_side = OrderSide.BUY
    else:  # short
        order_side = OrderSide.SELL
        # For short positions, swap take profit and stop loss prices
        take_profit_price, stop_loss_price = stop_loss_price, take_profit_price

    # Create stop loss and take profit requests
    stop_loss = StopLossRequest(stop_price=round(stop_loss_price, 2))
    take_profit = TakeProfitRequest(limit_price=round(take_profit_price, 2))

    # Create bracket order with stop loss and take profit
    bracket_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=shares,
        side=order_side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        stop_loss=stop_loss,
        take_profit=take_profit,
        extended_hours=extended_hours,
    )

    # Submit bracket order
    try:
        bracket_order = trading_client.submit_order(order_data=bracket_order_data)
        print(f"[BRACKET ORDER SUBMITTED] {symbol}: Order ID {bracket_order.id}")
    except APIError as e:
        print(f"[ERROR] Failed to submit bracket order for {symbol}: {str(e)}")
        return None

    # Track the order using OrderTracker
    await order_tracker.track_bracket_order(
        symbol=symbol,
        main_order_id=bracket_order.id,
        direction=direction,
        shares=shares,
    )

    return bracket_order


async def trade_single_stock_bracket(
    trading_client,
    symbol,
    equity_per_position,
    entry_price,
    take_profit_price,
    stop_loss_price,
    direction,
    extended_hours=False,
    exit_hour=14,
    exit_minute=59,
    debug=False,
):
    """
    Enhanced trading function with native Alpaca bracket orders and per-stock order tracking
    """

    # Initialize order tracker for this specific stock
    order_tracker = OrderTracker(trading_client, exit_hour, exit_minute, debug)

    print(f"\n=== ENHANCED TRADING: {symbol} ===")
    print(f"Direction: {direction}")
    print(f"Entry Price: ${entry_price:.2f}")
    print(f"Take Profit Target: ${take_profit_price:.2f}")
    print(f"Stop Loss: ${stop_loss_price:.2f}")
    print(f"Equity Allocation: ${equity_per_position:.2f}")
    print(f"EOD Exit: {exit_hour:02d}:{exit_minute:02d}")

    # Use native Alpaca bracket order with tracking
    bracket_order = await place_native_bracket_order(
        trading_client=trading_client,
        symbol=symbol,
        equity=equity_per_position,
        entry_price=entry_price,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        direction=direction,
        order_tracker=order_tracker,
        extended_hours=extended_hours,
    )
    if bracket_order is None:
        print(f"[ERROR] Failed to submit bracket order for {symbol}")
        return None

    # Wait for EOD monitoring to complete for this stock
    print(f"[{symbol}] Waiting for EOD monitoring to complete...")
    await order_tracker._start_eod_monitoring()
    print(f"[{symbol}] EOD monitoring completed")
