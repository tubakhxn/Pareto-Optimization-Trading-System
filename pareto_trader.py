import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# --- Data Generation ---
def generate_price_data(n=500, seed=42):
    np.random.seed(seed)
    returns = np.random.normal(0, 0.01, n)
    price = 100 * np.exp(np.cumsum(returns))
    return pd.Series(price, name="Price")

# --- Trading Strategy ---
def trading_strategy(price, ma_window, threshold):
    ma = price.rolling(ma_window).mean()
    signal = np.where(price > ma * (1 + threshold), 1, 0)
    signal = np.where(price < ma * (1 - threshold), -1, signal)
    signal = pd.Series(signal, index=price.index).fillna(0)
    returns = price.pct_change().fillna(0)
    strat_returns = signal.shift(1).fillna(0) * returns
    return strat_returns

# --- Metrics ---
def compute_metrics(strat_returns):
    total_return = np.sum(strat_returns)
    risk = np.var(strat_returns)
    drawdown = compute_max_drawdown(strat_returns)
    return total_return, risk, drawdown

def compute_max_drawdown(strat_returns):
    cum_returns = (1 + strat_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

# --- Pareto Front ---
def is_pareto_efficient(costs):
    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True  # Keep self
    return is_efficient

# --- Visualization ---
def plot_pareto_3d(results, pareto_mask):
    returns = results[:, 0]
    risks = results[:, 1]
    drawdowns = results[:, 2]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(returns.min(), returns.max())
    colors = cm.viridis(norm(returns))
    # Plot all points
    ax.scatter(returns, risks, drawdowns, c=colors, s=40, alpha=0.7, label='All Strategies')
    # Highlight Pareto front
    ax.scatter(returns[pareto_mask], risks[pareto_mask], drawdowns[pareto_mask],
               c='red', s=80, label='Pareto Front', edgecolor='k')
    # Surface-like smoothing (optional, for visual effect)
    from scipy.interpolate import griddata
    grid_x, grid_y = np.mgrid[returns.min():returns.max():30j, risks.min():risks.max():30j]
    grid_z = griddata((returns, risks), drawdowns, (grid_x, grid_y), method='cubic')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.viridis, alpha=0.2, linewidth=0, antialiased=True)
    ax.set_xlabel('Return')
    ax.set_ylabel('Risk (Variance)')
    ax.set_zlabel('Drawdown')
    ax.set_title('Pareto Front of Trading Strategies')
    ax.legend()
    plt.tight_layout()
    plt.show()

# --- Main ---
def main():
    price = generate_price_data()
    ma_windows = [5, 10, 20, 30, 50]
    thresholds = np.linspace(0.001, 0.02, 10)
    results = []
    params = []
    for ma in ma_windows:
        for th in thresholds:
            strat_returns = trading_strategy(price, ma, th)
            total_return, risk, drawdown = compute_metrics(strat_returns)
            results.append([total_return, risk, drawdown])
            params.append((ma, th))
    results = np.array(results)
    # For Pareto: maximize return, minimize risk/drawdown
    # So, flip sign of return for minimization
    costs = np.column_stack([-results[:, 0], results[:, 1], results[:, 2]])
    pareto_mask = is_pareto_efficient(costs)
    print("Best (Pareto optimal) strategies:")
    for i, (is_pareto, (ma, th), res) in enumerate(zip(pareto_mask, params, results)):
        if is_pareto:
            print(f"MA={ma}, Thresh={th:.4f} | Return={res[0]:.4f}, Risk={res[1]:.6f}, Drawdown={res[2]:.4f}")
    plot_pareto_3d(results, pareto_mask)

if __name__ == "__main__":
    main()
