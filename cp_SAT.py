
import logging as log
from ortools.sat.python import cp_model
from utils import get_distance_matrix


outcomes = [
	"UNKNOWN",
	"MODEL_INVALID",
	"FEASIBLE",
	"INFEASIBLE",
	"OPTIMAL",
]


def solve_OrTools(problem, verbose = True, timeout = 60, supplementary_constraint = False, enforce = False):
	dima = get_distance_matrix(problem)
	solver = cp_model.CpSolver()
	solver.parameters.max_time_in_seconds = timeout
	'''
	generate mip model using google or-tools and solve it

	:param dima: the distance matrix
	:return:  solution X, model, status
	'''
	if dima.ndim != 2 or dima.shape[0] != dima.shape[1]:
		raise ValueError("Invalid dima dimensions detected. Square matrix expected.")

	# determine number of nodes
	num_nodes = dima.shape[0]
	all_nodes = range(0, num_nodes)
	all_but_first_nodes = range(1, num_nodes)

	# Create the model.
	model = cp_model.CpModel()

	# generating decision variables X_ij
	if verbose :
		log.info(f'Creating {num_nodes * num_nodes} boolean x_ij variables... ')
	x = {}
	for i in all_nodes:
		for j in all_nodes:
			x[i, j] = model.NewBoolVar(f'x_i{i}j{j}')

	# constraint 1: leave every point exactly once
	if verbose :
		log.info(f'Creating {num_nodes} Constraint 1... ')
	for i in all_nodes:
		model.Add(sum(x[i,j] for j in all_nodes) == 1)


	# constraint 2: reach every point from exactly one other point
	if verbose :
		log.info(f'Creating {num_nodes} Constraint 2... ')
	for j in all_nodes:
		model.Add(sum(x[i,j] for i in all_nodes) == 1)


	# generating decision variables u_i for subtour elimination
	if verbose :
		log.info(f'Creating {num_nodes} boolean u_i variables... ')
	u = []
	for i in all_nodes:
		u.append(model.NewIntVar(0, num_nodes-1, f'u_i{i}'))

	# constraint 3.1: subtour elimination constraints (Miller-Tucker-Zemlin) part 1
	if verbose :
		log.info('Creating 1 Constraint 3.1... ')
	model.Add(u[0] == 0)


	# constraint 3.2: subtour elimination constraints (Miller-Tucker-Zemlin) part 2
	if verbose :
		log.info(f'Creating {len(all_but_first_nodes)} Constraint 3.2... ')
	for i in all_but_first_nodes:
		model.Add(1 <= u[i])
		model.Add(u[i] <= num_nodes)


	# constraint 3.3: subtour elimination constraints (Miller-Tucker-Zemlin) part 3
	if verbose :
		log.info(f'Creating {len(all_but_first_nodes)} Constraint 3.2... ')
	for i in all_but_first_nodes:
		for j in all_but_first_nodes:
			if enforce :
				model.Add(u[i] < u[j]).OnlyEnforceIf(x[i,j])
				if supplementary_constraint:
					model.Add(u[i] + 2 > u[j]).OnlyEnforceIf(x[i,j])

			else :
				model.Add(u[i] - u[j] + 1 <= num_nodes * (1 - x[i,j]))
				if supplementary_constraint:
					model.Add(u[i] - u[j] + 1 >= - num_nodes * (1 - x[i,j]))




	if verbose :
		log.info('Creating minimizing contraint')
	# Minimize the total distance
	model.Minimize(sum(dima[i,j] * x[i,j] for i in all_nodes for j in all_nodes))

	if verbose :
		log.info('Solving MIP model... ')
	status = solver.Solve(model)

	solution = [0] * dima.shape[0]
	for i, var in enumerate(u):
		solution[solver.Value(var)] = i # type: ignore
	path_length = solver.ObjectiveValue()

	return outcomes[status], solution, path_length #type: ignore