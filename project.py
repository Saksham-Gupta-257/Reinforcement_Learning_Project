import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

np.random.seed(42)

# Product, elasticity, base price, base demand, reference RL price.
products = [
    ('Samsung 24" HD', -0.5, 109.2, 80, 139.6),
    ('Samsung 55" 4K', -1.7, 674.3, 54, 636.9),
    ('Hisense 65" 4K', -1.1, 1412.1, 49, 971.0),
    ('Samsung 40" FHD', -0.7, 260.5, 67, 328.3),
    ('Samsung 49" 4K MU6290', -0.3, 444.7, 57, 811.6),
    ('Samsung 49" 4K Q6F', -4.4, 829.0, 97, 820.3),
    ('Samsung 50" FHD', -0.8, 418.4, 56, 324.4),
    ('Samsung 55" 4K Q8F', -8.4, 2011.6, 60, 1977.3),
    ('Samsung 65" 4K Q7F', -7.8, 2411.6, 60, 1253.6),
    ('Samsung 24" HD UN24H4500', -1.9, 142.7, 40, 119.3),
    ('Sony 40" FHD', -0.8, 423.8, 27, 329.4),
    ('Sony 43" 4K UHD', -5.6, 648.0, 154, 610.5),
    ('VIZIO 39" FHD', -1.8, 249.8, 59, 130.9),
    ('VIZIO 70" 4K XHDR', -6.5, 1300.0, 36, 1300.2),
]
df = pd.DataFrame(products, columns=[
    'Product Name', 'Elasticity', 'Base Price', 'Base Demand', 'RL Reference Price'
])

p0, e, ref = df['Base Price'], df['Elasticity'], df['RL Reference Price']
df['Unit Cost'] = np.clip(2 * ref - p0 * (1 - 1 / e), 0.15 * p0, 0.85 * p0)


class PricingEnv:
    def __init__(self, data, bins=250):
        self.df = data
        self.prices = []
        for _, row in data.iterrows():
            grid = np.linspace(0.45 * row['Base Price'], 2.05 * row['Base Price'], bins)
            grid[np.argmin(abs(grid - row['RL Reference Price']))] = row['RL Reference Price']
            self.prices.append(grid)

    def evaluate(self, i, price, day=None):
        r = self.df.iloc[i]
        demand = r['Base Demand'] * (1 + r['Elasticity'] *
                                    (price - r['Base Price']) / r['Base Price'])
        demand = max(0, demand * (0.98 if day == 0 else 1.02 if day == 1 else 1))
        revenue = price * demand
        return revenue - r['Unit Cost'] * demand, demand, revenue


class QLearning:
    def __init__(self, env, alpha=0.15, gamma=0.90, epsilon_decay=0.994):
        self.env, self.alpha, self.gamma = env, alpha, gamma
        self.epsilon, self.epsilon_decay = 1.0, epsilon_decay
        self.q = np.zeros((len(env.df), 2, len(env.prices[0])))
        self.history = []

    def train(self, episodes=1500):
        for _ in range(episodes):
            total = 0
            for i in range(len(self.env.df)):
                for day in (0, 1):
                    a = (np.random.randint(self.q.shape[2]) if np.random.rand() < self.epsilon
                         else np.argmax(self.q[i, day]))
                    reward, _, _ = self.env.evaluate(i, self.env.prices[i][a], day)
                    self.q[i, day, a] += self.alpha * (
                        reward + self.gamma * self.q[i, 1 - day].max() - self.q[i, day, a]
                    )
                    total += reward
            self.history.append(total / 2)  # average of the two day-type totals
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

    def best_prices(self):
        return [grid[np.argmax(self.q[i].mean(axis=0))]
                for i, grid in enumerate(self.env.prices)]


def make_results(env, agent):
    # Three methods: learned price, optimized continuous price, unchanged base price.
    rl_prices = agent.best_prices()
    scipy_prices = [minimize_scalar(
        lambda p, i=i: -env.evaluate(i, p)[0],
        bounds=(0.90 * r['Base Price'], 1.10 * r['Base Price']),
        method='bounded'
    ).x for i, r in env.df.iterrows()]
    fixed_prices = env.df['Base Price'].tolist()

    result = env.df[['Product Name', 'Elasticity', 'Base Price',
                     'Base Demand', 'Unit Cost']].copy()
    for label, prices in [('RL Optimal', rl_prices), ('Pure SciPy', scipy_prices),
                          ('Fixed', fixed_prices)]:
        metrics = np.array([env.evaluate(i, p) for i, p in enumerate(prices)])
        result[f'{label} Price ($)'] = np.round(prices, 1)
        result[f'{label} Demand'] = np.round(metrics[:, 1], 1)
        result[f'{label} Revenue ($)'] = np.round(metrics[:, 2], 2)
        result[f'{label} Profit ($)'] = np.round(metrics[:, 0], 2)
    return result


def draw_three_graphs(agent, result):
    methods = [('Q-Learning', 'RL Optimal Profit ($)', '#3673B7'),
               ('SciPy', 'Pure SciPy Profit ($)', '#1D996C'),
               ('Fixed Price', 'Fixed Profit ($)', '#EF962A')]
    fig = plt.figure(figsize=(19, 11))
    layout = fig.add_gridspec(2, 2, width_ratios=[1, 1.48], hspace=0.37, wspace=0.48)
    ax1, ax2, ax3 = (fig.add_subplot(layout[0, 0]),
                     fig.add_subplot(layout[:, 1]), fig.add_subplot(layout[1, 0]))

    # Graph 1: Q-Learning convergence (raw and 50-episode moving average).
    ax1.plot(agent.history, color='#A7B7CC', linewidth=1, label='Episode profit')
    ax1.plot(pd.Series(agent.history).rolling(50).mean(), color='#3673B7',
             linewidth=2.2, label='50-episode average')
    ax1.set(title='1. Q-Learning Convergence', xlabel='Episode',
            ylabel='Avg. profit across day types ($)')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    # Graph 2: all 14 products, compared on exactly the same profit model.
    y = np.arange(len(result))
    for offset, (name, column, color) in zip((-0.25, 0, 0.25), methods):
        ax2.barh(y + offset, result[column], height=0.24, label=name, color=color)
    ax2.set_yticks(y, result['Product Name'], fontsize=9)
    ax2.invert_yaxis()
    ax2.set(title='2. Profit Comparison for Each Product', xlabel='Profit ($)')
    ax2.legend(ncol=3, loc='lower right', fontsize=9)
    ax2.grid(axis='x', alpha=0.2)
    ax2.set_axisbelow(True)

    # Graph 3: final portfolio-profit comparison.
    totals = [result[col].sum() for _, col, _ in methods]
    bars = ax3.barh([m[0] for m in methods], totals,
                    color=[m[2] for m in methods], height=0.55)
    ax3.invert_yaxis()
    ax3.set_xlim(0, max(totals) * 1.30)
    ax3.set(title='3. Final Total-Profit Comparison', xlabel='Total profit ($)')
    ax3.grid(axis='x', alpha=0.2)
    ax3.set_axisbelow(True)
    for bar, value in zip(bars, totals):
        ax3.text(value + max(totals) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'${value:,.0f}', va='center', fontsize=10)

    fig.suptitle('Retail Pricing: Q-Learning vs SciPy vs Fixed Price',
                 fontsize=17, fontweight='bold')
    fig.savefig('pricing_three_graphs.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return methods


if __name__ == '__main__':
    env = PricingEnv(df)
    agent = QLearning(env)
    agent.train(episodes=1500)
    results = make_results(env, agent)

    columns = ['Product Name', 'Elasticity', 'Base Price', 'Base Demand']
    for name in ('RL Optimal', 'Pure SciPy', 'Fixed'):
        columns += [f'{name} Price ($)', f'{name} Demand', f'{name} Profit ($)']
    print('\n=== DYNAMIC PRICING RESULTS: Q-LEARNING vs SciPy vs FIXED PRICE ===')
    print(results[columns].to_string(index=False))

    print('\n--- Total Portfolio Profit ---')
    for name, col, _ in draw_three_graphs(agent, results):
        print(f'{name:15s}: ${results[col].sum():,.2f}')
    print('\nQ-Learning: alpha=0.15, gamma=0.90, epsilon=1.0 -> min 0.01, '
          'epsilon_decay=0.994, episodes=1500, bins=250')
    print('SciPy: minimize_scalar(method="bounded"), price range 45%-205% of base price')
    print('Fixed Price: original base price, no optimization')
    results[columns].to_csv('pricing_comparison_table.csv', index=False)
    print('\nSaved: pricing_three_graphs.png and pricing_comparison_table.csv')
