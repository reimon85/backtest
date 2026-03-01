from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """Base class for all strategies in the event-driven engine."""
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
    
    @abstractmethod
    def on_bar(self, bar: pd.Series):
        """Called for every bar (data point) in the backtest."""
        pass

class Portfolio:
    """Tracks positions, cash, and PnL in an event-driven backtest."""
    
    def __init__(self, initial_cash: float = 10000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {} # {symbol: quantity}
        self.equity_curve = [] # List of (timestamp, total_equity)
        self.history = [] # List of trades
        
    def buy(self, symbol: str, quantity: float, price: float, cost: float = 0.0):
        """Execute a buy order."""
        total_cost = quantity * price + cost
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            self.history.append({"type": "BUY", "symbol": symbol, "quantity": quantity, "price": price, "cost": cost})
            return True
        return False
        
    def sell(self, symbol: str, quantity: float, price: float, cost: float = 0.0):
        """Execute a sell order."""
        if self.positions.get(symbol, 0) >= quantity:
            self.cash += quantity * price - cost
            self.positions[symbol] -= quantity
            self.history.append({"type": "SELL", "symbol": symbol, "quantity": quantity, "price": price, "cost": cost})
            return True
        return False
        
    def update_equity(self, timestamp, current_prices: dict):
        """Calculate total equity (cash + position value) at a given point in time."""
        position_value = sum(quantity * current_prices.get(symbol, 0) 
                             for symbol, quantity in self.positions.items())
        total_equity = self.cash + position_value
        self.equity_curve.append({"timestamp": timestamp, "equity": total_equity})
        return total_equity

class ExecutionModel:
    """Simulates realistic execution by applying slippage and commissions."""
    
    def __init__(self, slippage_pct: float = 0.0, commission_pct: float = 0.0):
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        
    def execute(self, side: str, symbol: str, quantity: float, price: float):
        """Calculates execution price and total cost."""
        if side == "BUY":
            exec_price = price * (1 + self.slippage_pct)
            commission = (quantity * exec_price) * self.commission_pct
            return exec_price, commission
        elif side == "SELL":
            exec_price = price * (1 - self.slippage_pct)
            commission = (quantity * exec_price) * self.commission_pct
            return exec_price, commission
        return price, 0.0

class EventDrivenEngine:
    """The main engine that drives an event-driven backtest bar-by-bar."""
    
    def __init__(self, data: pd.DataFrame, strategy_class, execution_model=None, initial_cash: float = 10000.0):
        self.data = data
        self.portfolio = Portfolio(initial_cash)
        self.execution_model = execution_model or ExecutionModel()
        self.strategy = strategy_class(self.portfolio)
        
    def run(self, symbol: str = "Price"):
        """Run the simulation bar-by-bar."""
        for timestamp, bar in self.data.iterrows():
            # Current price for this bar (assuming closing price)
            current_price = bar[symbol]
            
            # Allow strategy to make decisions
            self.strategy.on_bar(bar)
            
            # Update portfolio equity at the end of the bar
            self.portfolio.update_equity(timestamp, {symbol: current_price})
            
        return pd.DataFrame(self.portfolio.equity_curve).set_index("timestamp")
