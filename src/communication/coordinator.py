import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

from .event_bus import EventBus, MarketEventMessage, AnalysisMessage, TradingSignalMessage, Channels
from ..models.events import MarketEvent, EnhancedMarketEvent
from ..models.analysis import AnalysisBundle
from ..models.trading import DebateResult, TradingSignal
from ..agents.monitoring_agent import InformationMonitoringAgent
from ..agents.background_agent import BackgroundEnhancementAgent
from ..agents.fundamental_agent import FundamentalAnalysisAgent
from ..agents.technical_agent import TechnicalAnalysisAgent
from ..agents.sentiment_agent import SentimentAnalysisAgent
from ..agents.macro_agent import MacroAnalysisAgent
from ..agents.debate_agents import DebateCoordinator
from ..agents.trading_agent import TradingAgent
from ..agents.execution_agent import ExecutionAgent
from ..config.settings import settings

logger = logging.getLogger(__name__)

class AgentCoordinator:
    """Coordinates all agents and manages the trading workflow"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.workflow_engine = WorkflowEngine()
        self.agents = {}
        self.is_running = False
        
    async def start(self):
        """Start all agents and coordination"""
        try:
            logger.info("Starting Agent Coordinator...")
            
            # Start event bus
            await self.event_bus.start()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Start workflow engine
            await self.workflow_engine.start(self.event_bus)
            
            # Set up message routing
            await self._setup_message_routing()
            
            # Start all agents
            await self._start_agents()
            
            # Start event bus listening
            asyncio.create_task(self.event_bus.start_listening())
            
            self.is_running = True
            logger.info("Agent Coordinator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Agent Coordinator: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop all agents and coordination"""
        logger.info("Stopping Agent Coordinator...")
        
        self.is_running = False
        
        # Stop all agents
        await self._stop_agents()
        
        # Stop workflow engine
        await self.workflow_engine.stop()
        
        # Stop event bus
        await self.event_bus.stop()
        
        logger.info("Agent Coordinator stopped")
    
    async def _initialize_agents(self):
        """Initialize all agents"""
        self.agents = {
            'monitoring': InformationMonitoringAgent(),
            'background': BackgroundEnhancementAgent(),
            'fundamental': FundamentalAnalysisAgent(),
            'technical': TechnicalAnalysisAgent(),
            'sentiment': SentimentAnalysisAgent(),
            'macro': MacroAnalysisAgent(),
            'debate': DebateCoordinator(),
            'trading': TradingAgent(),
            'execution': ExecutionAgent()
        }
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def _start_agents(self):
        """Start all agents"""
        for name, agent in self.agents.items():
            try:
                await agent.start()
                logger.info(f"Started {name} agent")
            except Exception as e:
                logger.error(f"Failed to start {name} agent: {e}")
                raise
    
    async def _stop_agents(self):
        """Stop all agents"""
        for name, agent in self.agents.items():
            try:
                await agent.stop()
                logger.info(f"Stopped {name} agent")
            except Exception as e:
                logger.error(f"Error stopping {name} agent: {e}")
    
    async def _setup_message_routing(self):
        """Set up message routing between agents"""
        
        # Market events -> Background enhancement
        await self.event_bus.subscribe(
            Channels.MARKET_EVENTS, 
            self._handle_market_event
        )
        
        # Enhanced events -> Analysis agents
        await self.event_bus.subscribe(
            "enhanced_events",
            self._handle_enhanced_event
        )
        
        # Analysis results -> Debate coordinator
        await self.event_bus.subscribe(
            Channels.ANALYSIS_RESULTS,
            self._handle_analysis_result
        )
        
        # Trading signals -> Execution agent
        await self.event_bus.subscribe(
            Channels.TRADING_SIGNALS,
            self._handle_trading_signal
        )
        
        logger.info("Message routing configured")
    
    async def _handle_market_event(self, message_data: Dict[str, Any]):
        """Handle market event message"""
        try:
            # Parse market event
            event_data = message_data.get('event_data', {})
            market_event = MarketEvent(**event_data)
            
            # Enhance event with background agent
            enhanced_event = await self.agents['background'].enhance_event(market_event)
            
            # Publish enhanced event
            enhanced_message = MarketEventMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                source="background_agent",
                event_data=enhanced_event.dict()
            )
            
            await self.event_bus.publish("enhanced_events", enhanced_message)
            
            logger.info(f"Enhanced market event {market_event.id}")
            
        except Exception as e:
            logger.error(f"Error handling market event: {e}")
    
    async def _handle_enhanced_event(self, message_data: Dict[str, Any]):
        """Handle enhanced event by running analysis"""
        try:
            # Parse enhanced event
            event_data = message_data.get('event_data', {})
            enhanced_event = EnhancedMarketEvent(**event_data)
            
            # Run all analyses in parallel
            analysis_tasks = [
                ('fundamental', self.agents['fundamental'].analyze(enhanced_event)),
                ('technical', self.agents['technical'].analyze(enhanced_event)),
                ('sentiment', self.agents['sentiment'].analyze(enhanced_event)),
                ('macro', self.agents['macro'].analyze(enhanced_event))
            ]
            
            # Execute analyses
            results = await asyncio.gather(
                *[task[1] for task in analysis_tasks],
                return_exceptions=True
            )
            
            # Create analysis bundle
            analysis_bundle = AnalysisBundle(event_id=enhanced_event.id)
            
            for i, (analysis_type, result) in enumerate(zip([task[0] for task in analysis_tasks], results)):
                if not isinstance(result, Exception):
                    setattr(analysis_bundle, analysis_type, result)
                    logger.info(f"Completed {analysis_type} analysis for event {enhanced_event.id}")
                else:
                    logger.error(f"Error in {analysis_type} analysis: {result}")
            
            # Publish analysis results
            analysis_message = AnalysisMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                source="analysis_coordinator",
                analysis_type="bundle",
                results=analysis_bundle.dict()
            )
            
            await self.event_bus.publish(Channels.ANALYSIS_RESULTS, analysis_message)
            
            logger.info(f"Published analysis bundle for event {enhanced_event.id}")
            
        except Exception as e:
            logger.error(f"Error handling enhanced event: {e}")
    
    async def _handle_analysis_result(self, message_data: Dict[str, Any]):
        """Handle analysis results by conducting debate"""
        try:
            # Parse analysis bundle
            results_data = message_data.get('results', {})
            analysis_bundle = AnalysisBundle(**results_data)
            
            # Conduct debate
            debate_result = await self.agents['debate'].conduct_debate(analysis_bundle)
            
            # Generate trading signal
            trading_signal = await self.agents['trading'].generate_trading_signal(
                debate_result, analysis_bundle
            )
            
            # Publish trading signal
            signal_message = TradingSignalMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                source="trading_agent",
                signal_data=trading_signal.dict()
            )
            
            await self.event_bus.publish(Channels.TRADING_SIGNALS, signal_message)
            
            logger.info(f"Generated trading signal: {trading_signal.action} {trading_signal.ticker}")
            
        except Exception as e:
            logger.error(f"Error handling analysis result: {e}")
    
    async def _handle_trading_signal(self, message_data: Dict[str, Any]):
        """Handle trading signal by executing trade"""
        try:
            # Parse trading signal
            signal_data = message_data.get('signal_data', {})
            trading_signal = TradingSignal(**signal_data)
            
            # Execute trade
            execution_result = await self.agents['execution'].execute_signal(trading_signal)
            
            # Log execution result
            if execution_result.success:
                logger.info(f"Successfully executed trade: {trading_signal.ticker}")
            else:
                logger.warning(f"Failed to execute trade: {execution_result.reason}")
            
        except Exception as e:
            logger.error(f"Error handling trading signal: {e}")
    
    async def process_market_event(self, event: MarketEvent) -> str:
        """Process a market event through the system"""
        
        # Generate workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Publish market event
        message = MarketEventMessage(
            id=workflow_id,
            timestamp=datetime.utcnow(),
            source="external",
            event_data=event.dict()
        )
        
        await self.event_bus.publish(Channels.MARKET_EVENTS, message)
        
        logger.info(f"Processing market event {event.id} with workflow {workflow_id}")
        return workflow_id
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            'coordinator_running': self.is_running,
            'agents': {}
        }
        
        for name, agent in self.agents.items():
            # Check if agent has status method
            if hasattr(agent, 'get_status'):
                status['agents'][name] = agent.get_status()
            else:
                status['agents'][name] = {'status': 'unknown'}
        
        return status

class WorkflowEngine:
    """Manages workflow execution and state"""
    
    def __init__(self):
        self.active_workflows = {}
        self.event_bus = None
        self.is_running = False
        
    async def start(self, event_bus: EventBus):
        """Start workflow engine"""
        self.event_bus = event_bus
        self.is_running = True
        
        # Start workflow monitoring
        asyncio.create_task(self._monitor_workflows())
        
        logger.info("Workflow Engine started")
    
    async def stop(self):
        """Stop workflow engine"""
        self.is_running = False
        logger.info("Workflow Engine stopped")
    
    async def start_workflow(self, workflow_type: str, params: Dict[str, Any]) -> str:
        """Start a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        workflow = {
            'id': workflow_id,
            'type': workflow_type,
            'params': params,
            'status': 'started',
            'created_at': datetime.utcnow(),
            'steps': [],
            'current_step': 0
        }
        
        self.active_workflows[workflow_id] = workflow
        
        logger.info(f"Started workflow {workflow_id} of type {workflow_type}")
        return workflow_id
    
    async def update_workflow_step(self, workflow_id: str, step_name: str, 
                                 status: str, data: Optional[Dict] = None):
        """Update workflow step"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            
            step = {
                'name': step_name,
                'status': status,
                'timestamp': datetime.utcnow(),
                'data': data or {}
            }
            
            workflow['steps'].append(step)
            workflow['current_step'] += 1
            
            if status == 'completed':
                workflow['status'] = 'running'
            elif status == 'failed':
                workflow['status'] = 'failed'
            
            logger.debug(f"Updated workflow {workflow_id} step {step_name}: {status}")
    
    async def complete_workflow(self, workflow_id: str, result: Dict[str, Any]):
        """Mark workflow as completed"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow['status'] = 'completed'
            workflow['completed_at'] = datetime.utcnow()
            workflow['result'] = result
            
            logger.info(f"Completed workflow {workflow_id}")
    
    async def _monitor_workflows(self):
        """Monitor active workflows for timeouts"""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                timeout_duration = 3600  # 1 hour timeout
                
                # Check for timed out workflows
                for workflow_id, workflow in list(self.active_workflows.items()):
                    if workflow['status'] in ['started', 'running']:
                        age = (current_time - workflow['created_at']).total_seconds()
                        
                        if age > timeout_duration:
                            workflow['status'] = 'timeout'
                            logger.warning(f"Workflow {workflow_id} timed out")
                
                # Clean up completed workflows older than 24 hours
                for workflow_id, workflow in list(self.active_workflows.items()):
                    if workflow['status'] in ['completed', 'failed', 'timeout']:
                        age = (current_time - workflow['created_at']).total_seconds()
                        
                        if age > 86400:  # 24 hours
                            del self.active_workflows[workflow_id]
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in workflow monitoring: {e}")
                await asyncio.sleep(300)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        return self.active_workflows.get(workflow_id)
    
    def get_active_workflows(self) -> Dict[str, Any]:
        """Get all active workflows"""
        return self.active_workflows.copy()