import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
import aiohttp
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, OrderSide, TimeInForce
from alpaca.trading.models import Order
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from ..models.trading import (
    TradingSignal, ExecutionResult, OrderRequest, ValidationResult,
    TakeProfitRequest, StopLossRequest, TradingAction
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

class ExecutionAgent:
    """Agent responsible for executing trades through broker API"""
    
    def __init__(self):
        self.alpaca_client = self._initialize_alpaca_client()
        self.market_data_client = self._initialize_market_data_client()
        self.order_manager = OrderManager()
        self.notification_service = NotificationService()
        
    def _initialize_alpaca_client(self) -> TradingClient:
        """Initialize Alpaca trading client"""
        return TradingClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            paper=settings.alpaca.paper_trading
        )
    
    def _initialize_market_data_client(self) -> StockHistoricalDataClient:
        """Initialize Alpaca market data client"""
        return StockHistoricalDataClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key
        )
    
    async def start(self):
        """Initialize the execution agent"""
        await self.order_manager.start()
        await self.notification_service.start()
        logger.info("Execution Agent started")
    
    async def stop(self):
        """Stop the execution agent"""
        await self.order_manager.stop()
        await self.notification_service.stop()
        logger.info("Execution Agent stopped")
    
    async def execute_signal(self, signal: TradingSignal) -> ExecutionResult:
        """Execute trading signal through broker API"""
        
        logger.info(f"Executing signal: {signal.action} {signal.quantity} {signal.ticker}")
        
        try:
            # Skip execution for HOLD signals
            if signal.action == TradingAction.HOLD:
                return ExecutionResult(
                    success=True,
                    reason="HOLD signal - no execution required"
                )
            
            # Validate market conditions
            market_validation = await self._validate_market_conditions(signal)
            if not market_validation.valid:
                return ExecutionResult(
                    success=False,
                    reason=market_validation.reason
                )
            
            # Create and submit order
            if signal.stop_loss > 0 and signal.take_profit > 0:
                # Use bracket order for stop loss and take profit
                order_response = await self._submit_bracket_order(signal)
            else:
                # Use simple market/limit order
                order_response = await self._submit_simple_order(signal)
            
            # Track order
            tracked_order = await self.order_manager.track_order(order_response)
            
            # Send notifications
            await self.notification_service.send_execution_notification(
                signal, order_response
            )
            
            return ExecutionResult(
                success=True,
                order_id=order_response.id,
                fill_price=float(order_response.filled_avg_price) if order_response.filled_avg_price else None,
                fill_time=order_response.filled_at,
                tracked_order=tracked_order
            )
            
        except Exception as e:
            logger.error(f"Execution failed for {signal.ticker}: {e}")
            await self.notification_service.send_error_notification(
                f"Execution failed for {signal.ticker}: {str(e)}"
            )
            return ExecutionResult(
                success=False,
                reason=f"Execution error: {str(e)}"
            )
    
    async def _validate_market_conditions(self, signal: TradingSignal) -> ValidationResult:
        """Validate current market conditions before execution"""
        
        try:
            # Check if market is open
            clock = self.alpaca_client.get_clock()
            if not clock.is_open:
                return ValidationResult(
                    valid=False,
                    reason="Market is closed"
                )
            
            # Check for halt conditions and get latest quote
            try:
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=[signal.ticker])
                latest_quotes = self.market_data_client.get_stock_latest_quote(quote_request)
                latest_quote = latest_quotes[signal.ticker]
                
                # Check if stock appears to be halted (no recent updates)
                if self._is_halted(latest_quote):
                    return ValidationResult(
                        valid=False,
                        reason=f"{signal.ticker} appears to be halted"
                    )
                
                # Check spread conditions
                if latest_quote.bid_price and latest_quote.ask_price:
                    spread_pct = (latest_quote.ask_price - latest_quote.bid_price) / latest_quote.bid_price
                    if spread_pct > 0.02:  # 2% spread threshold
                        return ValidationResult(
                            valid=False,
                            reason=f"Spread too wide: {spread_pct:.2%}"
                        )
                
            except Exception as e:
                logger.warning(f"Could not get latest quote for {signal.ticker}: {e}")
                # Continue with execution if quote check fails
            
            return ValidationResult(valid=True)
            
        except Exception as e:
            logger.error(f"Error validating market conditions: {e}")
            return ValidationResult(
                valid=False,
                reason=f"Market validation error: {str(e)}"
            )
    
    def _is_halted(self, quote) -> bool:
        """Check if stock appears to be halted"""
        # This is a simplified check
        # In production, you'd want more sophisticated halt detection
        try:
            if not quote.bid_price or not quote.ask_price:
                return True
            
            # Check if bid-ask spread is suspiciously wide (> 10%)
            if quote.ask_price > quote.bid_price * 1.1:
                return True
            
            return False
        except:
            return True  # Assume halted if we can't determine
    
    async def _submit_bracket_order(self, signal: TradingSignal) -> Order:
        """Submit bracket order with stop loss and take profit"""
        
        # Convert signal action to Alpaca OrderSide
        side = OrderSide.BUY if signal.action == TradingAction.BUY else OrderSide.SELL
        
        # Create bracket order request
        bracket_request = MarketOrderRequest(
            symbol=signal.ticker,
            qty=signal.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            take_profit={
                "limit_price": signal.take_profit
            },
            stop_loss={
                "stop_price": signal.stop_loss,
                "limit_price": signal.stop_loss
            }
        )
        
        # Submit order
        order = self.alpaca_client.submit_order(order_data=bracket_request)
        logger.info(f"Submitted bracket order: {order.id}")
        
        return order
    
    async def _submit_simple_order(self, signal: TradingSignal) -> Order:
        """Submit simple market order"""
        
        # Convert signal action to Alpaca OrderSide
        side = OrderSide.BUY if signal.action == TradingAction.BUY else OrderSide.SELL
        
        # Create market order request
        market_request = MarketOrderRequest(
            symbol=signal.ticker,
            qty=signal.quantity,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        
        # Submit order
        order = self.alpaca_client.submit_order(order_data=market_request)
        logger.info(f"Submitted market order: {order.id}")
        
        return order
    
    async def _submit_limit_order(self, signal: TradingSignal) -> Order:
        """Submit limit order at specified price"""
        
        # Convert signal action to Alpaca OrderSide
        side = OrderSide.BUY if signal.action == TradingAction.BUY else OrderSide.SELL
        
        # Create limit order request
        limit_request = LimitOrderRequest(
            symbol=signal.ticker,
            qty=signal.quantity,
            side=side,
            limit_price=signal.entry_price,
            time_in_force=TimeInForce.DAY
        )
        
        # Submit order
        order = self.alpaca_client.submit_order(order_data=limit_request)
        logger.info(f"Submitted limit order: {order.id}")
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            self.alpaca_client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current status of an order"""
        try:
            order = self.alpaca_client.get_order_by_id(order_id)
            return order
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

class OrderManager:
    """Manages order tracking and lifecycle"""
    
    def __init__(self):
        self.active_orders = {}
        self.is_running = False
        
    async def start(self):
        """Start order management"""
        self.is_running = True
        # Start order monitoring task
        asyncio.create_task(self._monitor_orders())
        logger.info("Order Manager started")
    
    async def stop(self):
        """Stop order management"""
        self.is_running = False
        logger.info("Order Manager stopped")
    
    async def track_order(self, order: Order) -> Dict[str, Any]:
        """Add order to tracking"""
        
        order_data = {
            'id': order.id,
            'symbol': order.symbol,
            'qty': order.qty,
            'side': order.side,
            'status': order.status,
            'created_at': order.created_at,
            'filled_at': order.filled_at,
            'filled_qty': order.filled_qty,
            'filled_avg_price': order.filled_avg_price
        }
        
        self.active_orders[order.id] = order_data
        logger.info(f"Tracking order: {order.id}")
        
        return order_data
    
    async def _monitor_orders(self):
        """Monitor active orders for status changes"""
        while self.is_running:
            try:
                # Check status of active orders
                for order_id in list(self.active_orders.keys()):
                    # This would check order status via API
                    # For now, we'll just clean up old orders
                    order_data = self.active_orders[order_id]
                    
                    # Remove completed orders after some time
                    if order_data.get('status') in ['filled', 'canceled', 'rejected']:
                        if order_id in self.active_orders:
                            del self.active_orders[order_id]
                            logger.info(f"Removed completed order from tracking: {order_id}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_active_orders(self) -> Dict[str, Any]:
        """Get all active orders"""
        return self.active_orders.copy()

class NotificationService:
    """Handles notifications for trading events"""
    
    def __init__(self):
        self.slack_webhook_url = settings.slack_webhook_url
        self.session = None
        
    async def start(self):
        """Start notification service"""
        self.session = aiohttp.ClientSession()
        logger.info("Notification Service started")
    
    async def stop(self):
        """Stop notification service"""
        if self.session:
            await self.session.close()
        logger.info("Notification Service stopped")
    
    async def send_execution_notification(self, signal: TradingSignal, order: Order):
        """Send notification for trade execution"""
        
        message = (
            f"🔄 **Trade Executed**\n"
            f"Action: {signal.action}\n"
            f"Symbol: {signal.ticker}\n"
            f"Quantity: {signal.quantity}\n"
            f"Price: ${signal.entry_price:.2f}\n"
            f"Order ID: {order.id}\n"
            f"Confidence: {signal.confidence:.1%}\n"
            f"Rationale: {signal.rationale}"
        )
        
        await self._send_slack_message(message)
        logger.info(f"Sent execution notification for {signal.ticker}")
    
    async def send_error_notification(self, error_message: str):
        """Send error notification"""
        
        message = f"❌ **Trading Error**\n{error_message}"
        await self._send_slack_message(message)
        logger.info("Sent error notification")
    
    async def send_signal_notification(self, signal: TradingSignal):
        """Send notification for generated signal"""
        
        if signal.action == TradingAction.HOLD:
            message = (
                f"⏸️ **Trading Signal: HOLD**\n"
                f"Symbol: {signal.ticker}\n"
                f"Reason: {signal.rationale}"
            )
        else:
            message = (
                f"📊 **Trading Signal Generated**\n"
                f"Action: {signal.action}\n"
                f"Symbol: {signal.ticker}\n"
                f"Quantity: {signal.quantity}\n"
                f"Entry: ${signal.entry_price:.2f}\n"
                f"Stop Loss: ${signal.stop_loss:.2f}\n"
                f"Take Profit: ${signal.take_profit:.2f}\n"
                f"Confidence: {signal.confidence:.1%}\n"
                f"R/R Ratio: {signal.risk_reward_ratio:.1f}\n"
                f"Rationale: {signal.rationale}"
            )
        
        await self._send_slack_message(message)
    
    async def _send_slack_message(self, message: str):
        """Send message to Slack webhook"""
        
        if not self.slack_webhook_url or not self.session:
            logger.warning("Slack webhook not configured or session not available")
            return
        
        try:
            payload = {
                "text": message,
                "username": "Trading Bot",
                "icon_emoji": ":robot_face:"
            }
            
            async with self.session.post(self.slack_webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.debug("Slack notification sent successfully")
                else:
                    logger.warning(f"Slack notification failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    async def send_daily_summary(self, trades_summary: Dict[str, Any]):
        """Send daily trading summary"""
        
        message = (
            f"📈 **Daily Trading Summary**\n"
            f"Total Trades: {trades_summary.get('total_trades', 0)}\n"
            f"Successful: {trades_summary.get('successful_trades', 0)}\n"
            f"Failed: {trades_summary.get('failed_trades', 0)}\n"
            f"Total P&L: ${trades_summary.get('total_pnl', 0):.2f}\n"
            f"Win Rate: {trades_summary.get('win_rate', 0):.1%}"
        )
        
        await self._send_slack_message(message)