#!/usr/bin/env python3
"""
Multi-Agent LLM Trading System
Main application entry point

This is the main entry point for the multi-agent trading system.
It coordinates all agents and manages the overall trading workflow.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

from src.communication.coordinator import AgentCoordinator
from src.models.events import MarketEvent, EventType, EventSeverity
from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TradingSystemManager:
    """Main manager for the trading system"""
    
    def __init__(self):
        self.coordinator = AgentCoordinator()
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
    async def start(self):
        """Start the trading system"""
        try:
            logger.info("=" * 60)
            logger.info("Starting Multi-Agent LLM Trading System")
            logger.info("=" * 60)
            
            # Validate configuration
            await self._validate_configuration()
            
            # Start the coordinator
            await self.coordinator.start()
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.is_running = True
            logger.info("Trading system started successfully!")
            logger.info("System is now monitoring markets and ready to trade...")
            
            # Start example market event processing
            if settings.environment == "development":
                asyncio.create_task(self._generate_example_events())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading system"""
        if not self.is_running:
            return
            
        logger.info("Shutting down trading system...")
        
        self.is_running = False
        
        # Stop the coordinator
        await self.coordinator.stop()
        
        logger.info("Trading system shutdown complete")
    
    async def _validate_configuration(self):
        """Validate system configuration before startup"""
        
        logger.info("Validating system configuration...")
        
        # Check required API keys
        required_keys = []
        
        if not settings.openai.api_key or settings.openai.api_key == "your-openai-api-key-here":
            required_keys.append("OPENAI_API_KEY")
        
        if not settings.alpaca.api_key or settings.alpaca.api_key == "your-alpaca-api-key-here":
            required_keys.append("ALPACA_API_KEY")
        
        if not settings.alpaca.secret_key or settings.alpaca.secret_key == "your-alpaca-secret-key-here":
            required_keys.append("ALPACA_SECRET_KEY")
        
        if required_keys:
            logger.warning("Missing required API keys:")
            for key in required_keys:
                logger.warning(f"  - {key}")
            logger.warning("Please update your .env file with valid API keys")
            
            if not settings.simulation_mode:
                raise ValueError("Cannot start without required API keys. Set SIMULATION_MODE=true to run without real APIs.")
        
        # Validate paper trading settings
        if settings.alpaca.paper_trading:
            logger.info("Running in PAPER TRADING mode - no real money at risk")
        else:
            logger.warning("LIVE TRADING mode enabled - real money at risk!")
            
        logger.info("Configuration validation complete")
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    async def _generate_example_events(self):
        """Generate example market events for development/testing"""
        
        logger.info("Starting example event generation (development mode)")
        
        # Wait a bit for system to fully start
        await asyncio.sleep(10)
        
        example_events = [
            MarketEvent(
                id="example_1",
                timestamp=datetime.utcnow(),
                type=EventType.NEWS,
                severity=EventSeverity.MEDIUM,
                ticker="AAPL",
                title="Apple Reports Strong Q4 Earnings",
                description="Apple Inc. reported better-than-expected quarterly earnings with strong iPhone sales and services revenue growth.",
                source="example_news",
                url="https://example.com/apple-earnings",
                metadata={
                    "sector": "Technology",
                    "market_cap": "large",
                    "earnings_surprise": 0.15
                }
            ),
            MarketEvent(
                id="example_2",
                timestamp=datetime.utcnow(),
                type=EventType.PRICE_MOVEMENT,
                severity=EventSeverity.HIGH,
                ticker="TSLA",
                title="Tesla Stock Surges on Autonomous Driving News",
                description="Tesla shares jump 8% after announcing breakthrough in full self-driving technology and regulatory approval timeline.",
                source="market_data",
                metadata={
                    "price_change_pct": 8.2,
                    "volume_spike": 3.5,
                    "sector": "Electric Vehicles"
                }
            )
        ]
        
        for i, event in enumerate(example_events):
            # Process event through the system
            workflow_id = await self.coordinator.process_market_event(event)
            logger.info(f"Generated example event {i+1}: {event.title} (workflow: {workflow_id})")
            
            # Wait between events
            await asyncio.sleep(30)
    
    async def process_market_event(self, event: MarketEvent) -> str:
        """Process a market event through the system"""
        return await self.coordinator.process_market_event(event)
    
    def get_system_status(self) -> dict:
        """Get current system status"""
        return {
            'running': self.is_running,
            'timestamp': datetime.utcnow().isoformat(),
            'agents': self.coordinator.get_agent_status() if self.coordinator else {},
            'environment': settings.environment,
            'paper_trading': settings.alpaca.paper_trading,
            'simulation_mode': settings.simulation_mode
        }

async def main():
    """Main application entry point"""
    
    try:
        # Create and start the trading system
        trading_system = TradingSystemManager()
        await trading_system.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure we're using the event loop
    if sys.platform == "win32":
        # Windows-specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the main application
    asyncio.run(main())