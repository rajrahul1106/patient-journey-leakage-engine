# Project context

Patient Journey Leakage & Intervention Optimization Engine.
Identifies where and why patients leave a treatment journey, then
recommends the most profitable intervention.

## Rules
- Read config/config.yaml and DECISIONS.md before writing any code.
- Never hardcode a parameter that exists in config.yaml.
- Never invent an assumption. If something is undefined, stop and ask me.
- Censoring is the core correctness requirement. A patient without enough
  observation time is censored, never a drop-off.
- Build only the module I ask for. Do not scaffold ahead.
- Plain pandas, numpy, statsmodels, matplotlib. No frameworks.