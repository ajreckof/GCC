import numpy as np
from utils import get_distance_matrix, print_solution


def ACO(problem, n_fourmis=50, n_iterations=100, alpha=2, beta=1, taux_evaporation=0.5, Q=1, Tmin=0, Tmax=1, verbose = True):
	distance = get_distance_matrix(problem)
	#On initialise le tableau des phéromones pour chaque edge à 1 et on fait un best_path le pire possible pour initialiser
	n_points = len(distance)
	pheromone = np.ones((n_points, n_points))*Tmax
	meilleur_chemin = None
	meilleur_chemin_long = np.inf
	meilleur_chemin_fourmi=[]
	meilleur_chemin_episode=[]
	#A chaque iteration, on recherche les chemins donc on vide chemins et chemins_long
	for iteration in range(n_iterations):
		chemins = []
		chemins_long = []
		#Pour chaque fourmi, on la lance sur un point aléatoire et met tous ses points en non visités
		for ant in range(n_fourmis):
			nonvisite = [True]*n_points
			point_actuel = np.random.randint(n_points)
			nonvisite[point_actuel] = False
			chemin = [point_actuel]
			chemin_long = 0

			#Tant que tout n'est pas visité, on continue de tirer un chemin parmi les non visités
			while True in nonvisite:
				nonvisite_indices = np.where(nonvisite)[0]
				probabilites = np.zeros(len(nonvisite_indices))
				#Ici, la fameuse partie haute de la formule des probas [τC(vj)]^α.[ηC(vj)]^β
				for i, point_nonvisite in enumerate(nonvisite_indices):
					probabilites[i] = pheromone[point_actuel, point_nonvisite]**alpha / distance[point_actuel,point_nonvisite]**beta
				#divisé par leur somme donc
				probabilites /= np.sum(probabilites)

				#On choisit le point en f° des probas
				point_suivant = np.random.choice(nonvisite_indices, p=probabilites)
				chemin.append(point_suivant)
				chemin_long += distance[point_actuel,point_suivant]
				nonvisite[point_suivant] = False
				point_actuel = point_suivant

			chemin_long += distance[chemin[-1], chemin[0]]
			#Une fois le chemin prêt, on le garde s'il est moins long que les autres.
			chemins.append(chemin)
			chemins_long.append(chemin_long)

			if chemin_long < meilleur_chemin_long:
				meilleur_chemin = chemin
				meilleur_chemin_long = chemin_long
			meilleur_chemin_fourmi+=[meilleur_chemin_long]
		meilleur_chemin_episode+=[meilleur_chemin_long]
		#On évapore les phéromones et on les met à jour
		pheromone *= taux_evaporation
		for chemin, chemin_long in zip(chemins, chemins_long):
			for i in range(n_points-1):
				pheromone[chemin[i], chemin[i+1]] += Q/chemin_long
			pheromone[chemin[-1], chemin[0]] += Q/chemin_long
		#On applique les bornes Tmin et Tmax
		pheromone = np.clip(pheromone, Tmin, Tmax)

	return "FEASIBLE", meilleur_chemin, meilleur_chemin_long