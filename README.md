# Dynamic Retail Pricing Using Q-Learning

A Reinforcement Learning project that uses **Q-Learning** to find profitable selling prices for 14 television products.

## Algorithms Used

- **Q-Learning:** Learns optimal pricing decisions using rewards.
- **SciPy Optimization:** Finds profitable prices using bounded mathematical optimization.
- **Fixed Pricing:** Uses the original base price of each product.

## Installation

Install the required Python libraries:

`pip install numpy pandas scipy matplotlib`

## How to Run

Run the following command:

`python project.py`

## Q-Learning Parameters

| Parameter | Value |
|---|---|
| Learning Rate (α) | 0.15 |
| Discount Factor (γ) | 0.90 |
| Initial Epsilon | 1.0 |
| Epsilon Decay | 0.994 |
| Episodes | 1500 |
| Price Actions | 250 |

## Output

The program generates three graphs:

1. Q-Learning Training Convergence
2. Profit Comparison for Each Product
3. Final Total-Profit Comparison

It also saves CSV files containing product-level results and first-episode Q-value updates.

## Research Paper

[Dynamic Retail Pricing via Q-Learning (2024)](https://arxiv.org/pdf/2411.18261)
