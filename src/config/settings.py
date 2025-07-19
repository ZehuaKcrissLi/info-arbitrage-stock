import os
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig(BaseModel):
    host: str
    port: int
    database: str
    username: str
    password: str

class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    password: Optional[str] = None

class AlpacaConfig(BaseModel):
    api_key: str
    secret_key: str
    base_url: str
    paper_trading: bool = True

class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    max_tokens: int

class MonitoringConfig(BaseModel):
    news_sources: list
    social_sources: list
    update_frequency: int

class RiskConfig(BaseModel):
    max_position_pct: float
    max_total_exposure: float
    max_sector_exposure: float
    max_loss_pct: float
    risk_reward_ratio: float
    atr_stop_multiplier: float
    max_daily_loss: float
    max_consecutive_losses: int
    cooldown_period: int

class TradingConfig(BaseModel):
    paper_trading: bool
    max_daily_trades: int
    max_concurrent_positions: int
    analysis_interval: int
    monitoring_interval: int
    news_check_interval: int

class Settings:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._substitute_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _substitute_env_vars(self):
        """Substitute environment variables in configuration"""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            else:
                return obj
        
        self._config = substitute_recursive(self._config)
    
    def get(self, key: str, default=None):
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def database(self) -> DatabaseConfig:
        db_config = self._config['database']['postgresql']
        return DatabaseConfig(**db_config)
    
    @property
    def redis(self) -> RedisConfig:
        redis_config = self._config['database']['redis']
        return RedisConfig(**redis_config)
    
    @property
    def alpaca(self) -> AlpacaConfig:
        alpaca_config = self._config['apis']['alpaca']
        alpaca_config['paper_trading'] = self._config['trading']['execution']['paper_trading']
        return AlpacaConfig(**alpaca_config)
    
    @property
    def llm(self) -> LLMConfig:
        llm_config = self._config['agents']['llm']
        return LLMConfig(**llm_config)
    
    @property
    def monitoring(self) -> MonitoringConfig:
        monitoring_config = self._config['agents']['monitoring']
        return MonitoringConfig(**monitoring_config)
    
    @property
    def risk(self) -> RiskConfig:
        risk_config = {
            **self._config['risk_management']['position_limits'],
            **self._config['risk_management']['risk_parameters'],
            **self._config['risk_management']['circuit_breaker']
        }
        return RiskConfig(**risk_config)
    
    @property
    def trading(self) -> TradingConfig:
        trading_config = {
            **self._config['trading']['execution'],
            **self._config['trading']['scheduling']
        }
        return TradingConfig(**trading_config)
    
    @property
    def finnhub_api_key(self) -> str:
        return self._config['apis']['data_providers']['finnhub']['api_key']
    
    @property
    def newsapi_key(self) -> str:
        return self._config['apis']['data_providers']['newsapi']['api_key']
    
    @property
    def openai_api_key(self) -> str:
        return self._config['apis']['data_providers']['openai']['api_key']
    
    @property
    def anthropic_api_key(self) -> str:
        return self._config['apis']['data_providers']['anthropic']['api_key']
    
    @property
    def slack_webhook_url(self) -> str:
        return self._config['monitoring']['alerts']['slack']['webhook_url']

# Global settings instance
settings = Settings()