import numpy as np
from utils import get_distance_matrix, print_solution
from functools import partial
from multiprocessing import Pool
import time
from IPython.display import display
from seaborn import histplot
import matplotlib.pyplot as plt

def ant_chemin(n_points, pheromone, distance, alpha, beta, *args):
	nonvisite = [True]*n_points
	point_actuel = np.random.randint(n_points)
	nonvisite[point_actuel] = False
	chemin = [point_actuel]
	chemin_long : float = 0

	#Tant que tout n'est pas visité, on continue de tirer un chemin parmi les non visités
	while True in nonvisite:
		nonvisite_indices = np.where(nonvisite)[0]
		probabilites = np.zeros(len(nonvisite_indices))
		
		#Ici, la fameuse partie haute de la formule des probas [τC(vj)]^α.[ηC(vj)]^β
		for i, point_nonvisite in enumerate(nonvisite_indices):
			probabilites[i] = pheromone[point_actuel, point_nonvisite]**alpha / distance[point_actuel,point_nonvisite]**beta
			
		#Divisé par leur somme donc
		probabilites /= np.sum(probabilites)
		
		#On choisit le point en f° des probas
		point_suivant = np.random.choice(nonvisite_indices, p=probabilites)
		chemin.append(point_suivant)
		chemin_long += distance[point_actuel,point_suivant]
		nonvisite[point_suivant] = False
		point_actuel = point_suivant
	
	# On cloture la boucle en revenant au point de départ
	chemin_long += distance[chemin[-1], chemin[0]]
	
	return chemin, chemin_long

def ACO(problem, n_fourmis=50, n_iterations=200, alpha=2, beta=2, taux_evaporation=0.5, Q=1, Tmin=1, Tmax=20, verbose = True, n_process = None, time_out = 60):
	
	#On initialise le tableau des phéromones pour chaque edge à 1 et on fait un best_path le pire possible pour initialiser
	distance = get_distance_matrix(problem)
	n_points = problem.dimension

	#on rectifie Q pour prendre en compte le nombre de points et de fourmis
	Q *= n_points/n_fourmis


	pheromone = np.ones((n_points, n_points))*Tmax
	meilleur_chemin = None
	meilleur_chemin_long = np.mean(distance)
	start_time = time.time()
	#A chaque iteration, on recherche les chemins donc on vide chemins et chemins_long
	with Pool(n_process) as pool :
		while time.time() < start_time + time_out :
			chemins = []
			chemins_long = []
			func = partial(ant_chemin, n_points, pheromone, distance, alpha, beta)
			#Pour chaque fourmi, on la lance sur un point aléatoire et met tous ses points en non visités
			for chemin, chemin_long in pool.imap_unordered(func,range(n_fourmis)):
				chemins.append(chemin)
				chemins_long.append(chemin_long)

				#Une fois le chemin prêt, on le garde s'il est moins long que les autres.
				if chemin_long < meilleur_chemin_long:
					meilleur_chemin = chemin
					meilleur_chemin_long = chemin_long
			#On évapore les phéromones
			pheromone *= taux_evaporation
			
			# on mets à jour les pheromones
			for chemin, chemin_long in zip(chemins, chemins_long):
				for i in range(n_points-1):
					pheromone[chemin[i], chemin[i+1]] += meilleur_chemin_long * Q/chemin_long
				pheromone[chemin[-1], chemin[0]] += meilleur_chemin_long * Q/chemin_long

			#On applique les bornes Tmin et Tmax
			pheromone = np.clip(pheromone, Tmin, Tmax)

	return "FEASIBLE", meilleur_chemin, meilleur_chemin_long
