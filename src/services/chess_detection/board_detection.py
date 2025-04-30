from typing import List
import numpy as np
import cv2
import random
from ultralytics.engine.results import Results
from deap import base, creator, tools, algorithms

from common import models_manager


class GeneticBoard:
    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(np.uint8) * 255
        self.width = mask.shape[1]
        self.height = mask.shape[0]
        self.pop_size = 35
        self.num_generations = 120
        self.cxpb = 0.7
        self.mutpb = 0.3
        self.ind_size = 8

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            self.generate_individual,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=0,
            up=max(self.width, self.height),
            indpb=0.1,
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def generate_individual(self):
        return [0] * self.ind_size

    def create_quadrilateral_mask(self, individual):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        points = np.array(
            [(individual[i], individual[i + 1]) for i in range(0, self.ind_size, 2)],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
        return mask

    def cost_function(self, individual):
        quad_mask = self.create_quadrilateral_mask(individual)
        xor_result = cv2.bitwise_xor(quad_mask, self.mask)
        return (np.count_nonzero(xor_result),)

    def evaluate(self, individual):
        return self.cost_function(individual)

    def find_initial_corners(self):
        edges = cv2.Canny(self.mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        intersections = []
        if lines is not None:
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    rho1, theta1 = lines[i][0]
                    rho2, theta2 = lines[j][0]
                    A = np.array(
                        [
                            [np.cos(theta1), np.sin(theta1)],
                            [np.cos(theta2), np.sin(theta2)],
                        ]
                    )
                    b = np.array([[rho1], [rho2]])
                    try:
                        x0, y0 = np.linalg.solve(A, b)
                        x0, y0 = int(np.round(x0)), int(np.round(y0))
                        if 0 <= x0 <= self.width and 0 <= y0 <= self.height:
                            intersections.append((x0, y0))
                    except np.linalg.LinAlgError:
                        pass
        corners = []
        corners_float = cv2.cornerHarris(self.mask, blockSize=2, ksize=3, k=0.04)
        corners_float = cv2.dilate(corners_float, None)
        ret, corners_float = cv2.threshold(
            corners_float, 0.01 * corners_float.max(), 255, 0
        )
        corners_float = np.uint8(corners_float)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners_float)
        if centroids is not None:
            for centroid in centroids:
                x, y = int(centroid[0]), int(centroid[1])
                corners.append((x, y))
        corners.extend(intersections)
        return corners

    def initialize_population(self):
        corners = self.find_initial_corners()
        population = []
        for _ in range(self.pop_size):
            if len(corners) >= 4:
                selected_corners = random.sample(corners, 4)
                individual = [coord for point in selected_corners for coord in point]
                population.append(creator.Individual(individual))
            else:
                population.append(creator.Individual(self.generate_individual()))
        return population

    def run_ga(self):
        pop = self.initialize_population()
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, logbook = algorithms.eaMuPlusLambda(
            pop,
            self.toolbox,
            mu=self.pop_size,
            lambda_=self.pop_size * 2,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.num_generations,
            stats=stats,
            halloffame=hof,
            verbose=False,
        )
        return hof[0]

    def get_best_coordinates(self):
        best_individual = self.run_ga()
        return [
            (best_individual[i], best_individual[i + 1])
            for i in range(0, self.ind_size, 2)
        ]


def get_corners(image: np.ndarray) -> List[int]:
    model = models_manager.get_yolo_model("chess_board")

    results: List[Results] = model.predict(source=image, save=False, device="cpu")

    mask = results[0].masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    coords_estimator = GeneticBoard(mask)
    best_coords = coords_estimator.get_best_coordinates()

    return np.array(best_coords)
