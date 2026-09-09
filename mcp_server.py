"""
MCP Server for Particle Filter Monte Carlo Localization Skill
"""

import json
import sys
from client import ParticleFilter1D

pf = ParticleFilter1D(num_particles=100)

def handle_call(name: str, args: dict) -> dict:
    if name == "step_filter":
        vel = args.get("velocity", 1.0)
        meas = args.get("measurement", 0.0)
        mean, std = pf.step(vel, meas)
        return {"estimated_x": mean, "uncertainty_std": std}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
