import numpy as np

def total_cost(y_true, scores, amounts, threshold, fp_cost=5.0):
    y = np.asarray(y_true)
    a = np.asarray(amounts)
    flagged = np.asarray(scores) >= threshold
    missed = (~flagged) & (y == 1)         # frauds we let through
    false_alarm = flagged & (y == 0)       # legit transactions we flagged
    return a[missed].sum() + fp_cost * false_alarm.sum()

def best_threshold(y_true, scores, amounts, fp_cost=5.0):
    grid = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y_true, scores, amounts, t, fp_cost) for t in grid]
    i = int(np.argmin(costs))
    return grid[i], costs[i], grid, costs