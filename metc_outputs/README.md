# METC output folder check

Use `metc_simulation_report.html` from this folder. This is the Koen-corrected output.

Quick checks that identify the correct report:

- Scenario grid: **5 acute toxicity scenarios × 3 EWOC modes × 2 burn-in settings = 30 combinations**.
- Simulations per combination: **200**.
- Correct acute prior skeleton: **0.004, 0.021, 0.066, 0.150, 0.266**.
- Correct subacute prior skeleton: **0.012, 0.036, 0.084, 0.157, 0.250**.
- If a report shows **6 toxicity scenarios / 36 combinations** or acute prior **0.021, 0.066, 0.150, 0.266, 0.396**, it is an old output folder and should not be used.

Regenerate this folder with:

```bash
python metc_simulation_report.py --n-sim 200
```
