import networkx as nx
import numpy as np
import tsplib95
import random
import matplotlib.pyplot as plt
import time
from utils import get_distance_matrix as compute_euclidean_distance_matrix

"""Simple Travelling Salesperson Problem (TSP) on a circuit board."""

import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp



def create_data_model(problem):
    """Stores the data for the problem."""
    data = {}
    data["num_vehicles"] = 1
    data["depot"] = 0
    data["distances"] = compute_euclidean_distance_matrix(problem)
    return data

def incompletOR(problem, verbose):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(problem)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        data["distances"].shape[0]-1, data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = data["distances"].astype(int)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)


    #Résultat
    chemin=[0]
    route_distance=0
    index = routing.Start(0)
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        chemin+=[index]
        route_distance += data["distances"][previous_index][index]
    route_distance += data["distances"][index][0]
    return "FEASIBLE",chemin,route_distance

    