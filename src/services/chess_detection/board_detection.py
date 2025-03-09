from typing import List
import numpy as np
import cv2
import random
from ultralytics.engine.results import Results
from deap import base, creator, tools, algorithms

from common import models_manager


class GeneticTrapezoid:
    def __init__(self, mask: MatLike):
        self.mask = mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)  # Assume the largest contour is the main one

        # Get convex hull
        hull = cv2.convexHull(main_contour).flatten().tolist()
        hull = np.array([[hull[i], hull[i + 1]] for i in range(0, len(hull), 2)], dtype=np.int32)

        # --- Find the largest quadrilateral within the convex hull ---
        def largest_quadrilateral(hull):
            """Finds the quadrilateral with the largest area within the convex hull."""
            max_area = 0
            best_quad = None

            for i in range(len(hull)):
                for j in range(i + 1, len(hull)):
                    for k in range(j + 1, len(hull)):
                        for l in range(k + 1, len(hull)):
                            quad = np.array([hull[i], hull[j], hull[k], hull[l]])
                            area = cv2.contourArea(quad)
                            if area > max_area:
                                max_area = area
                                best_quad = quad
            return best_quad

        self.initial_trapezoid = largest_quadrilateral(hull)

    def find_trapezoid(
            self, 
            population_size = 25,
            generations = 100,
            crossover_prob = 0.9,
            mutation_prob = 0.3,
            num_elites = 2,
        ) -> List[Tuple[int]]:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        def generate_individual():
            """Generates an individual: 4 (x, y) coordinates from the largest quadrilateral."""
            return creator.Individual(self.initial_trapezoid.tolist())

        toolbox.register("individual", generate_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # --- Evaluation Function ---
        def evaluate(individual):
            """Evaluates the fitness of an individual (quadrilateral)."""
            polygon = np.array(individual, dtype=np.int32)

            # Initialize the polygon mask with zeros, same shape as `mask`
            polygon_mask = np.zeros_like(self.mask, dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [polygon], 255)

            # XOR and count non-zero pixels
            xor_result = cv2.bitwise_xor(self.mask, polygon_mask)

            cv2.imshow("INTENTO", polygon_mask)
            cv2.imshow("MODELO", self.mask)
            cv2.imshow("RESULTADO", xor_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            score = cv2.countNonZero(xor_result) / (self.mask.shape[0] * self.mask.shape[1])

            return score,

        toolbox.register("evaluate", evaluate)

        # --- Crossover (Single-Point) ---
        def cxSinglePoint(ind1, ind2):
            """Executes a single-point crossover on the coordinates of the individuals."""
            cxpoint = random.randint(1, len(ind1) - 1)
            ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:].copy(), ind1[cxpoint:].copy()
            return ind1, ind2

        toolbox.register("mate", cxSinglePoint)

        # --- Mutation (Adaptive) ---
        def mutAdaptive(individual, indpb, generation, max_generations):
            """Mutates an individual with adaptive mutation strength."""
            # Decrease mutation strength over time
            max_mutation = 50  # Initial maximum displacement
            min_mutation = 5   # Final minimum displacement
            mutation_range = max_mutation - (max_mutation - min_mutation) * (generation / max_generations)

            for i in range(len(individual)):
                if random.random() < indpb:
                    dx = random.randint(-int(mutation_range), int(mutation_range))
                    dy = random.randint(-int(mutation_range), int(mutation_range))

                    individual[i][0] = max(0, min(self.mask.shape[1], individual[i][0] + dx))
                    individual[i][1] = max(0, min(self.mask.shape[0], individual[i][1] + dy))

            return individual,

        toolbox.register("mutate", mutAdaptive, indpb=0.7)

        # --- Selection ---
        toolbox.register("select", tools.selTournament, tournsize=3)

        def best_polygon():
            pop = toolbox.population(n=population_size)
            hof = tools.HallOfFame(num_elites)

            # Evaluate the initial population
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # Run the genetic algorithm
            for g in range(generations):
                # Select the next generation
                offspring = toolbox.select(pop, len(pop) - num_elites)
                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))

                # Apply crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < crossover_prob:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Apply mutation
                for mutant in offspring:
                    if random.random() < mutation_prob:
                        toolbox.mutate(mutant, generation=g, max_generations=generations)
                        del mutant.fitness.values

                # Evaluate individuals with invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Add elite individuals to the offspring
                offspring.extend(hof.items)

                # Update the HallOfFame
                hof.update(pop)

                # Replace the old population with the new offspring
                pop[:] = offspring

                # Gather statistics
                fits = [ind.fitness.values[0] for ind in pop]

            # Get the best individual
            best_ind = hof[0]

            return best_ind, (1 - best_ind.fitness.values[0])
        
        return best_polygon()
    
def get_corners(image: np.ndarray) -> List[int]:
    model = models_manager.get_yolo_model("chess_board")

    results: List[Results] = model.predict(source=image, save=False, device="cpu")

    mask = results[0].masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    cv2.imshow("m", mask)

    coords_estimator = GeneticTrapezoid(mask)
    best_coords = coords_estimator.find_trapezoid()

    return np.array(best_coords).flatten().tolist()
