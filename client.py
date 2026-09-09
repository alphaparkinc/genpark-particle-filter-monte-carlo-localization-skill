"""
Particle Filter Monte Carlo Localization Skill Client
Pure Python Standard Library implementation of Sequential Importance Resampling (SIR) Particle Filter (Thrun et al.).
Estimates multimodal, non-Gaussian continuous agent positions through distributed weighted particle clouds.
"""

from typing import List, Dict, Any, Tuple, Optional
import random
import math


class Particle:
    def __init__(self, x: float, weight: float = 1.0):
        self.x = x
        self.weight = weight


class ParticleFilter1D:
    def __init__(self, num_particles: int = 100, x_range: Tuple[float, float] = (0.0, 100.0), seed: Optional[int] = 42):
        self.num_particles = num_particles
        self.rng = random.Random(seed)
        # Uniform initial distribution
        self.particles: List[Particle] = [
            Particle(self.rng.uniform(x_range[0], x_range[1]), 1.0 / num_particles)
            for _ in range(num_particles)
        ]

    def predict(self, velocity: float, noise_std: float = 0.5):
        """Move particles forward with motion noise."""
        for p in self.particles:
            p.x += velocity + self.rng.gauss(0, noise_std)

    def update_weights(self, measurement: float, sensor_noise_std: float = 1.0):
        """Update particle weights based on Gaussian likelihood."""
        total_weight = 0.0
        for p in self.particles:
            diff = measurement - p.x
            # Gaussian likelihood
            prob = math.exp(- (diff ** 2) / (2.0 * (sensor_noise_std ** 2)))
            p.weight = prob
            total_weight += prob

        # Normalize weights
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
        else:
            uniform_w = 1.0 / self.num_particles
            for p in self.particles:
                p.weight = uniform_w

    def resample(self):
        """Low-variance systematic resampling."""
        weights = [p.weight for p in self.particles]
        new_particles = []
        r = self.rng.uniform(0, 1.0 / self.num_particles)
        c = weights[0]
        i = 0

        for m in range(self.num_particles):
            u = r + m * (1.0 / self.num_particles)
            while u > c and i < self.num_particles - 1:
                i += 1
                c += weights[i]
            new_particles.append(Particle(self.particles[i].x, 1.0 / self.num_particles))

        self.particles = new_particles

    def estimate_state(self) -> Tuple[float, float]:
        """Compute weighted mean and standard deviation."""
        mean = sum(p.x * p.weight for p in self.particles)
        var = sum(p.weight * ((p.x - mean) ** 2) for p in self.particles)
        return mean, math.sqrt(var)

    def step(self, velocity: float, measurement: float) -> Tuple[float, float]:
        self.predict(velocity)
        self.update_weights(measurement)
        self.resample()
        return self.estimate_state()
