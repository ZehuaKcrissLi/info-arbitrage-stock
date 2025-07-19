import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
import redis.asyncio as redis
from pydantic import BaseModel

from ..config.settings import settings

logger = logging.getLogger(__name__)

class BaseMessage(BaseModel):
    """Base class for all messages"""
    id: str
    timestamp: datetime
    type: str
    source: str
    
class EventBus:
    """Redis-based event bus for inter-agent communication"""
    
    def __init__(self):
        self.redis_client = None
        self.subscribers = {}
        self.event_history = EventHistory()
        self.is_running = False
        
    async def start(self):
        """Initialize the event bus"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.is_running = True
            await self.event_history.start()
            
            logger.info("Event Bus started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Event Bus: {e}")
            raise
    
    async def stop(self):
        """Stop the event bus"""
        self.is_running = False
        
        if self.redis_client:
            await self.redis_client.close()
        
        await self.event_history.stop()
        logger.info("Event Bus stopped")
    
    async def publish(self, channel: str, message: BaseMessage):
        """Publish message to channel"""
        if not self.is_running or not self.redis_client:
            logger.warning("Event bus not running, cannot publish message")
            return
        
        try:
            # Serialize message
            serialized_message = message.json()
            
            # Publish to Redis
            await self.redis_client.publish(channel, serialized_message)
            
            # Store in event history
            await self.event_history.store(channel, message)
            
            logger.debug(f"Published message to {channel}: {message.type}")
            
        except Exception as e:
            logger.error(f"Failed to publish message to {channel}: {e}")
    
    async def subscribe(self, channel: str, callback: Callable):
        """Subscribe to channel with callback"""
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        
        self.subscribers[channel].append(callback)
        logger.info(f"Subscribed to channel: {channel}")
    
    async def unsubscribe(self, channel: str, callback: Callable):
        """Unsubscribe from channel"""
        if channel in self.subscribers:
            try:
                self.subscribers[channel].remove(callback)
                if not self.subscribers[channel]:
                    del self.subscribers[channel]
                logger.info(f"Unsubscribed from channel: {channel}")
            except ValueError:
                logger.warning(f"Callback not found for channel: {channel}")
    
    async def start_listening(self):
        """Start listening for messages"""
        if not self.is_running or not self.redis_client:
            logger.warning("Event bus not running, cannot start listening")
            return
        
        try:
            # Create pubsub instance
            pubsub = self.redis_client.pubsub()
            
            # Subscribe to all channels with registered callbacks
            for channel in self.subscribers.keys():
                await pubsub.subscribe(channel)
                logger.info(f"Listening on channel: {channel}")
            
            # Start listening loop
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    await self._handle_message(message)
                    
        except Exception as e:
            logger.error(f"Error in event bus listening: {e}")
    
    async def _handle_message(self, redis_message):
        """Handle incoming message from Redis"""
        try:
            channel = redis_message['channel']
            data = json.loads(redis_message['data'])
            
            # Call all subscribers for this channel
            callbacks = self.subscribers.get(channel, [])
            
            for callback in callbacks:
                try:
                    # Call callback asynchronously
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                        
                except Exception as e:
                    logger.error(f"Error in callback for channel {channel}: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def get_message_history(self, channel: str, limit: int = 100) -> List[Dict]:
        """Get message history for a channel"""
        return await self.event_history.get_history(channel, limit)

class EventHistory:
    """Stores event history for debugging and replay"""
    
    def __init__(self):
        self.redis_client = None
        self.history_key_prefix = "event_history:"
        self.max_history_per_channel = 1000
        
    async def start(self):
        """Initialize event history"""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password,
                decode_responses=True
            )
            logger.info("Event History started")
        except Exception as e:
            logger.error(f"Failed to start Event History: {e}")
    
    async def stop(self):
        """Stop event history"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Event History stopped")
    
    async def store(self, channel: str, message: BaseMessage):
        """Store message in history"""
        if not self.redis_client:
            return
        
        try:
            history_key = f"{self.history_key_prefix}{channel}"
            
            # Store message with timestamp
            message_data = {
                'timestamp': message.timestamp.isoformat(),
                'data': message.dict()
            }
            
            # Add to list (newest first)
            await self.redis_client.lpush(history_key, json.dumps(message_data))
            
            # Trim list to max size
            await self.redis_client.ltrim(history_key, 0, self.max_history_per_channel - 1)
            
        except Exception as e:
            logger.error(f"Failed to store event history: {e}")
    
    async def get_history(self, channel: str, limit: int = 100) -> List[Dict]:
        """Get message history for channel"""
        if not self.redis_client:
            return []
        
        try:
            history_key = f"{self.history_key_prefix}{channel}"
            
            # Get messages (newest first)
            messages = await self.redis_client.lrange(history_key, 0, limit - 1)
            
            # Parse and return
            parsed_messages = []
            for msg in messages:
                try:
                    parsed_messages.append(json.loads(msg))
                except json.JSONDecodeError:
                    continue
            
            return parsed_messages
            
        except Exception as e:
            logger.error(f"Failed to get event history: {e}")
            return []

class MessageRouter:
    """Routes messages between agents based on rules"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.routing_rules = {}
        
    async def start(self):
        """Start message router"""
        # Subscribe to all channels for routing
        await self.event_bus.subscribe("*", self._route_message)
        logger.info("Message Router started")
    
    async def stop(self):
        """Stop message router"""
        logger.info("Message Router stopped")
    
    def add_routing_rule(self, source_channel: str, target_channels: List[str], 
                        condition: Optional[Callable] = None):
        """Add routing rule"""
        self.routing_rules[source_channel] = {
            'targets': target_channels,
            'condition': condition
        }
        logger.info(f"Added routing rule: {source_channel} -> {target_channels}")
    
    async def _route_message(self, message_data: Dict):
        """Route message based on rules"""
        try:
            message_type = message_data.get('type')
            source = message_data.get('source')
            
            # Find applicable routing rules
            for source_pattern, rule in self.routing_rules.items():
                if self._matches_pattern(source, source_pattern):
                    # Check condition if specified
                    if rule['condition'] and not rule['condition'](message_data):
                        continue
                    
                    # Route to target channels
                    for target_channel in rule['targets']:
                        await self.event_bus.publish(target_channel, message_data)
                        
        except Exception as e:
            logger.error(f"Error routing message: {e}")
    
    def _matches_pattern(self, source: str, pattern: str) -> bool:
        """Check if source matches pattern"""
        if pattern == "*":
            return True
        return source == pattern

# Message types for different events
class MarketEventMessage(BaseMessage):
    """Message for market events"""
    type: str = "market_event"
    event_data: Dict[str, Any]

class AnalysisMessage(BaseMessage):
    """Message for analysis results"""
    type: str = "analysis_result"
    analysis_type: str
    results: Dict[str, Any]

class TradingSignalMessage(BaseMessage):
    """Message for trading signals"""
    type: str = "trading_signal"
    signal_data: Dict[str, Any]

class SystemMessage(BaseMessage):
    """Message for system events"""
    type: str = "system_event"
    event_type: str
    details: Dict[str, Any]

# Channel names
class Channels:
    """Standard channel names"""
    MARKET_EVENTS = "market_events"
    ANALYSIS_RESULTS = "analysis_results"
    TRADING_SIGNALS = "trading_signals"
    SYSTEM_EVENTS = "system_events"
    RISK_ALERTS = "risk_alerts"
    EXECUTION_RESULTS = "execution_results"
    AGENT_STATUS = "agent_status"