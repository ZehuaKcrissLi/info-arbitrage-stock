import asyncio
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiohttp
import numpy as np
from sentence_transformers import SentenceTransformer

from ..models.events import MarketEvent, EnhancedMarketEvent, HistoricalEvent
from ..config.settings import settings

logger = logging.getLogger(__name__)

class BackgroundEnhancementAgent:
    """RAG-based agent that enriches market events with contextual information"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = VectorStore()
        self.web_searcher = WebSearchTool()
        self.knowledge_base = FinancialKnowledgeBase()
        self.session = None
        
    async def start(self):
        """Initialize the agent"""
        self.session = aiohttp.ClientSession()
        await self.vector_store.initialize()
        logger.info("Background Enhancement Agent started")
    
    async def stop(self):
        """Stop the agent"""
        if self.session:
            await self.session.close()
        logger.info("Background Enhancement Agent stopped")
    
    async def enhance_event(self, event: MarketEvent) -> EnhancedMarketEvent:
        """Add context and background to market event"""
        try:
            # Retrieve relevant historical context
            historical_context = await self._retrieve_historical_context(event)
            
            # Search for additional information
            web_context = await self._search_web_context(event)
            
            # Query internal knowledge base
            kb_context = await self._query_knowledge_base(event)
            
            # Generate enhanced context using LLM
            enhanced_context = await self._generate_enhanced_context(
                event, historical_context, web_context, kb_context
            )
            
            return EnhancedMarketEvent(
                original_event=event,
                historical_context=historical_context,
                web_context=web_context,
                knowledge_context=kb_context,
                enhanced_summary=enhanced_context,
                confidence_score=self._calculate_enhancement_confidence(
                    historical_context, web_context, kb_context
                )
            )
            
        except Exception as e:
            logger.error(f"Error enhancing event {event.id}: {e}")
            # Return minimal enhancement on error
            return EnhancedMarketEvent(
                original_event=event,
                historical_context="No historical context available",
                web_context="No web context available", 
                knowledge_context="No knowledge context available",
                enhanced_summary=event.summary,
                confidence_score=0.1
            )
    
    async def _retrieve_historical_context(self, event: MarketEvent) -> str:
        """Retrieve similar historical events from vector store"""
        try:
            # Create query embedding
            query_text = f"{event.headline} {event.summary} {' '.join(event.tickers)}"
            query_embedding = self.embedding_model.encode(query_text)
            
            # Search for similar events
            similar_events = await self.vector_store.similarity_search(
                query_embedding, k=5
            )
            
            if not similar_events:
                return "No similar historical events found."
            
            # Format historical context
            context_parts = []
            for event_data in similar_events:
                impact = event_data.get('market_impact', 0)
                impact_desc = "positive" if impact > 0 else "negative" if impact < 0 else "neutral"
                
                context_parts.append(
                    f"• {event_data.get('description', 'Unknown event')} "
                    f"({event_data.get('timestamp', 'Unknown date')}) - "
                    f"Market impact: {impact_desc} ({impact:.2f})"
                )
            
            return "Similar historical events:\n" + "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error retrieving historical context: {e}")
            return "Error retrieving historical context."
    
    async def _search_web_context(self, event: MarketEvent) -> str:
        """Search web for additional context"""
        try:
            search_queries = self._generate_search_queries(event)
            search_results = []
            
            for query in search_queries[:3]:  # Limit to 3 queries
                results = await self.web_searcher.search(query, limit=2)
                search_results.extend(results)
            
            if not search_results:
                return "No additional web context found."
            
            # Summarize search results
            context_parts = []
            for result in search_results[:5]:  # Limit to 5 results
                context_parts.append(
                    f"• {result.get('title', 'Unknown')}: {result.get('snippet', 'No description')}"
                )
            
            return "Recent web information:\n" + "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error searching web context: {e}")
            return "Error retrieving web context."
    
    async def _query_knowledge_base(self, event: MarketEvent) -> str:
        """Query internal knowledge base"""
        try:
            # Query for company information
            company_info = []
            for ticker in event.tickers:
                info = await self.knowledge_base.get_company_info(ticker)
                if info:
                    company_info.append(f"• {ticker}: {info}")
            
            # Query for event type context
            event_context = await self.knowledge_base.get_event_context(event.event_type)
            
            context_parts = []
            if company_info:
                context_parts.append("Company background:\n" + "\n".join(company_info))
            if event_context:
                context_parts.append(f"Event context: {event_context}")
            
            return "\n\n".join(context_parts) if context_parts else "No knowledge base context available."
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return "Error querying knowledge base."
    
    def _generate_search_queries(self, event: MarketEvent) -> List[str]:
        """Generate search queries for web search"""
        queries = []
        
        # Company-specific queries
        for ticker in event.tickers:
            queries.append(f"{ticker} {event.event_type.value} latest news")
            queries.append(f"{ticker} stock price impact {event.event_type.value}")
        
        # Event-specific queries
        if event.event_type.value in ["earnings", "merger", "fda_approval"]:
            queries.append(f"{event.event_type.value} impact stock market historical")
        
        # Recent news query
        queries.append(f"{' '.join(event.tickers)} {event.headline[:50]}")
        
        return queries
    
    async def _generate_enhanced_context(
        self, 
        event: MarketEvent, 
        historical_context: str, 
        web_context: str, 
        kb_context: str
    ) -> str:
        """Generate enhanced context using LLM"""
        try:
            # This is a simplified version - in production you'd use OpenAI/Anthropic
            # For now, we'll create a structured summary
            
            summary_parts = [
                f"Event: {event.headline}",
                f"Type: {event.event_type.value}",
                f"Tickers: {', '.join(event.tickers)}",
                f"Sentiment: {'Positive' if event.sentiment > 0 else 'Negative' if event.sentiment < 0 else 'Neutral'}",
                "",
                "Enhanced Context:",
                historical_context,
                "",
                web_context,
                "",
                kb_context
            ]
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating enhanced context: {e}")
            return f"Enhanced summary: {event.summary}"
    
    def _calculate_enhancement_confidence(
        self, 
        historical_context: str, 
        web_context: str, 
        kb_context: str
    ) -> float:
        """Calculate confidence score for the enhancement"""
        score = 0.0
        
        # Base score
        score += 0.2
        
        # Historical context quality
        if "similar historical events" in historical_context.lower():
            score += 0.3
        elif "no similar" not in historical_context.lower():
            score += 0.1
        
        # Web context quality
        if "recent web information" in web_context.lower():
            score += 0.3
        elif "no additional" not in web_context.lower():
            score += 0.1
        
        # Knowledge base quality
        if "company background" in kb_context.lower():
            score += 0.2
        elif "no knowledge" not in kb_context.lower():
            score += 0.1
        
        return min(1.0, score)

class VectorStore:
    """Simple vector store for historical events"""
    
    def __init__(self):
        self.events = []
        self.embeddings = []
        self.index = None
        
    async def initialize(self):
        """Initialize the vector store"""
        # Load historical events if available
        await self._load_historical_events()
        logger.info(f"Vector store initialized with {len(self.events)} events")
    
    async def _load_historical_events(self):
        """Load historical events from storage"""
        # This would load from a database in production
        # For now, we'll create some sample events
        sample_events = [
            {
                "id": "hist_1",
                "timestamp": "2024-01-15",
                "tickers": ["AAPL"],
                "event_type": "earnings",
                "description": "Apple reports strong Q1 earnings beat",
                "market_impact": 0.8,
                "price_change_1h": 2.5,
                "price_change_1d": 3.2
            },
            {
                "id": "hist_2", 
                "timestamp": "2024-02-10",
                "tickers": ["MSFT"],
                "event_type": "earnings",
                "description": "Microsoft misses revenue expectations",
                "market_impact": -0.6,
                "price_change_1h": -1.8,
                "price_change_1d": -2.1
            }
        ]
        
        self.events = sample_events
    
    async def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar events"""
        if not self.events:
            return []
        
        # Simple similarity search (in production, use FAISS or similar)
        # For now, return sample similar events
        return self.events[:k]
    
    async def add_event(self, event: HistoricalEvent):
        """Add event to vector store"""
        event_dict = {
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "tickers": event.tickers,
            "event_type": event.event_type,
            "description": event.description,
            "market_impact": event.market_impact,
            "price_change_1h": event.price_change_1h,
            "price_change_1d": event.price_change_1d
        }
        
        self.events.append(event_dict)
        logger.info(f"Added event to vector store: {event.id}")

class WebSearchTool:
    """Tool for searching web content"""
    
    def __init__(self):
        self.session = None
    
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """Search web for query"""
        try:
            # This is a placeholder implementation
            # In production, you'd use Google Custom Search API, Bing API, etc.
            
            # For now, return mock results
            mock_results = [
                {
                    "title": f"Search result for: {query}",
                    "snippet": f"Mock search result snippet for query: {query}",
                    "url": f"https://example.com/search?q={query}"
                }
            ]
            
            return mock_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []

class FinancialKnowledgeBase:
    """Internal knowledge base for financial information"""
    
    def __init__(self):
        # This would be backed by a database in production
        self.company_data = {
            "AAPL": "Apple Inc. - Technology company known for iPhone, iPad, Mac computers",
            "MSFT": "Microsoft Corporation - Software and cloud computing company",
            "GOOGL": "Alphabet Inc. - Search engine and technology conglomerate",
            "AMZN": "Amazon.com Inc. - E-commerce and cloud computing giant",
            "TSLA": "Tesla Inc. - Electric vehicle and clean energy company"
        }
        
        self.event_contexts = {
            "earnings": "Quarterly earnings reports can significantly impact stock prices",
            "merger": "Merger and acquisition announcements often cause significant price movements",
            "fda_approval": "FDA approvals for pharmaceutical companies can cause major price swings",
            "analyst_upgrade": "Analyst upgrades typically lead to positive price movements",
            "analyst_downgrade": "Analyst downgrades typically lead to negative price movements"
        }
    
    async def get_company_info(self, ticker: str) -> Optional[str]:
        """Get company information"""
        return self.company_data.get(ticker)
    
    async def get_event_context(self, event_type: str) -> Optional[str]:
        """Get context for event type"""
        return self.event_contexts.get(event_type)