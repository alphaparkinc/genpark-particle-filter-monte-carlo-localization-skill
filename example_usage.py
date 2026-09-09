"""
Demonstration of Particle Filter Monte Carlo Localization Skill
"""

from client import ParticleFilter1D

def main():
    print("=== Agent Localization with 1D SIR Particle Filter ===")
    pf = ParticleFilter1D(num_particles=200, x_range=(0.0, 50.0))

    # True agent motion: starts at 10.0, moves +2.0 each step
    true_position = 10.0
    velocity = 2.0

    print("Initial Particle Mean:", pf.estimate_state()[0])

    print("\nSimulating 5 steps of motion and noisy sensor updates:")
    for step_num in range(1, 6):
        true_position += velocity
        noisy_measurement = true_position + (step_num % 2 * 0.4 - 0.2)
        est_x, std_x = pf.step(velocity, noisy_measurement)
        print(f"  Step {step_num}: True Pos={true_position:5.2f} | Particle Estimate={est_x:5.2f} (StdDev={std_x:.3f})")

    final_est, final_std = pf.estimate_state()
    assert abs(final_est - true_position) < 1.0
    print("\nParticle Filter Monte Carlo Localization Verification PASS!")

if __name__ == "__main__":
    main()
