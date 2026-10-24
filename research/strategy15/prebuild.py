"""Build the outcome and random-picker caches once (so parallel runs do not race)."""
import sys
from harness import load, placebo
ctx = load()
ex, h = sys.argv[1], int(sys.argv[2])
placebo(ctx, ex, h)
print("done", ex, h)
