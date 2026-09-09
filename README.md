# GenPark Particle Filter Monte Carlo Localization Skill

Sequential Importance Resampling (SIR) particle filter for non-linear Monte Carlo state estimation and localization.

Discover more at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph TD
    A[Particle Cloud {x_i, w_i}] --> B[Motion Update: Predict x_i += v + noise]
    B --> C[Observation Likelihood Weighting w_i = P(z|x_i)]
    C --> D[Weight Normalization]
    D --> E[Systematic Low-Variance Resampling]
    E --> F[New Equi-weighted Particle Cloud]
```

## Features
- Non-parametric Monte Carlo representation of arbitrary belief states.
- Low-variance systematic resampling wheel algorithm.
- Pure Python standard library.
