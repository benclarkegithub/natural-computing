import numpy as np


class ParticleSwarm:
    """ A simulation of a particle swarm """
    particles = None
    fitnesses = None
    local_bests = None
    global_best = {'position': None, 'value': None}
    iteration = 0

    def __init__(self, n_particles, parameters, dimensions, limit, fitness, velocity, K):
        """
        Takes number of particles, the parameters, the number of dimensions,
        the limit (search space is [-limit, +limit]), and a fitness function.
        """
        self.n_particles = n_particles
        self.inertia = parameters['inertia']
        self.alpha_1 = parameters['alpha_1']
        self.alpha_2 = parameters['alpha_2']
        self.dimensions = dimensions
        self.limit = limit
        self.fitness = fitness
        self.velocity_min = velocity['min']
        self.velocity_max = velocity['max']
        self.K = K

        # Initialise the particles
        self.particles = (np.random.rand(n_particles, dimensions) * 2 * limit) - limit
        self.velocities = np.zeros(n_particles)[:, None]
        self.fitnesses = fitness(self.particles)
        self.local_bests = self.particles
        self.global_best['position'] = self.particles[self.fitnesses.argmin()]
        self.global_best['value'] = self.fitnesses.min()

    def iterate(self):
        self.update_velocities()
        self.update_positions()
        self.get_fitnesses()
        self.update_local_bests()
        self.update_global_best()
        self.iteration += 1

    def update_velocities(self):
        r = np.random.rand(2, self.n_particles, self.dimensions)
        v1 = self.inertia * self.velocities
        v2 = self.alpha_1 * r[0] * (self.local_bests - self.particles)
        v3 = self.alpha_2 * r[1] * (self.global_best['position'] - self.particles)
        # Velocity control (constriction)
        K_i = self.K ** (self.iteration % 20)
        self.velocities = K_i * (v1 + v2 + v3)
        # Velocity control (clamping)
        if (self.iteration != 0) and ((self.velocity_min != 0) or (self.velocity_max != np.inf)):
            total_velocities = np.sqrt((self.velocities ** 2).sum(axis=1))
            if not np.any(total_velocities == 0):
                if self.velocity_min != 0:
                    less_than_velocity_min = np.logical_and(-(K_i * self.velocity_min) < total_velocities, total_velocities < (K_i * self.velocity_min))
                    speed_up_factor = np.where(less_than_velocity_min, self.velocity_min / total_velocities, 1)
                    self.velocities = speed_up_factor[:, None] * self.velocities
                if self.velocity_max != np.inf:
                    speed_down_factor = np.where((K_i * self.velocity_max) < total_velocities, self.velocity_max / total_velocities, 1)
                    self.velocities = speed_down_factor[:, None] * self.velocities

    def update_positions(self):
        self.particles = self.particles + self.velocities
        # After updating the particles' positions, they might be outside the search space
        # If a particle is outside the search space (in a particular dimension), reverse its
        # speed in that dimension and set its position to the boundary.
        outside_search_space = np.logical_or(self.particles < -self.limit, self.particles > self.limit)
        self.velocities = np.where(outside_search_space, -self.velocities, self.velocities)
        self.particles = np.where(self.particles < -self.limit, -self.limit, self.particles)
        self.particles = np.where(self.particles > self.limit, self.limit, self.particles)

    def get_fitnesses(self):
        self.fitnesses = self.fitness(self.particles)

    def update_local_bests(self):
        best_fitnesses = self.fitness(self.local_bests)
        self.local_bests = np.where(self.fitnesses < best_fitnesses, self.particles, self.local_bests)

    def get_global_best(self):
        return self.global_best

    def update_global_best(self):
        if self.fitnesses.min() < self.global_best['value']:
            self.global_best['position'] = self.particles[self.fitnesses.argmin()]
            self.global_best['value'] = self.fitnesses.min()

    def get_iteration(self):
        return self.iteration


# Sphere function
def sphere():
    return lambda x: (x ** 2).sum(axis=1)[:, None]


# Rastrigin function
def rastrigin(D):
    return lambda x: ((10 * D) + ((x ** 2) - (10 * np.cos(2 * np.pi * x))).sum(axis=1))[:, None]
