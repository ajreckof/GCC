
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import logging as log
import networkx as nx
from math import inf
import numpy as np
import tsplib95
import os
import re
import sys
import time



# configure logger for info level
log.basicConfig(
	format='%(asctime)s %(levelname)-8s %(message)s',
	level=log.INFO,
	datefmt='%Y-%m-%d %H:%M:%S',
	stream= sys.stdout)

def min_max_from_value(value, padding = .1):
	m = min(value)
	M = max(value)
	d = M-m
	return m - padding * d, M + padding * d

def is_iterable(value):
	try:
		iter(value)
		return True
	except TypeError:
		return False



def get_evenly_spaced_colors(N, saturation = 1.0, value= 1.0):
	# Calculate equally spaced hue values
	hues = np.linspace(0, 1, N, endpoint= False)
	saturations = np.full_like(hues, saturation)
	values = np.full_like(hues, value)

	# Stack the arrays as columns
	hsv = np.column_stack((hues, saturations, values))

	# Convert each hue to RGB and returns it 
	return hsv_to_rgb(hsv)

def offset_line_in_display(point1, point2, transform, offset=0.1):
	# Calculate the direction of the line connecting the two points

	point1_display, point2_display = transform.transform((point1,point2))
	dx, dy = point2_display[0] - point1_display[0], point2_display[1] - point1_display[1]
	
	if  (dx,dy) < (0,0):
		perp_direction = np.array([dy,-dx])
	else : 
		perp_direction = np.array([-dy,dx])


	offset_perp_direction = perp_direction * offset / (np.linalg.norm(perp_direction))

	point1_offset_display = point1_display + offset_perp_direction
	point2_offset_display = point2_display + offset_perp_direction

	point1_offset, point2_offset = transform.inverted().transform([point1_offset_display, point2_offset_display])

	# Calculate offsets for each line
	return point1_offset, point2_offset


def plot_offset_lines(ax, points, number_of_lines, line_index, total_linewidth = 3, **kwargs):
	plt.xlim(min_max_from_value(points[:,0]))
	plt.ylim(min_max_from_value(points[:,1]))
	transform = ax.transData
	line_width = total_linewidth / number_of_lines
	offset = 1.5 * line_width * (0.5 + line_index - number_of_lines / 2) 
	for i in range(len(points) - 1):
		ax.plot(
			*zip(*offset_line_in_display(points[i], points[i+1], transform, offset= offset)), 
			linewidth= line_width, 
			**kwargs
		)
	
	label_to_handles = { label : handle for handle, label in zip(*ax.get_legend_handles_labels()) }
	ax.legend(handles=label_to_handles.values(), labels=label_to_handles.keys())  


def print_solution(problem, solutions, name_to_label, **kwargs):
	points = get_positions(problem)
	colors = get_evenly_spaced_colors(len(solutions))

	#Dessin du résultat
	fig = plt.figure(figsize=(15,15))
	fig.suptitle(problem.name)
	ax = fig.add_subplot()
	number_of_solutions = len(solutions)

	for i in range(len(points)):
		ax.annotate(str(i + 1), points[i,:])

	for i, (color, (name, solution)) in enumerate(zip(colors,solutions.items())):
		if solution :
			solution.append(solution[0])
			ordered_points = points[solution]
			plot_offset_lines(ax, ordered_points, number_of_solutions, i , c=color, linestyle='-', label = name_to_label(name), **kwargs)

	ax.scatter(points[:,0], points[:,1], c='k', marker='o', zorder=2)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	plt.show()

def pivot_and_plot(df, max_n = 4, value = 'time'):
	if(value == "len"):
		for problemName in df["problem"].unique():
			bestSol=check_best_solution(problemName)
			if(bestSol[0]):
				df.loc[-1]= ["solution",problemName,bestSol[1], 0]
				df.index = df.index + 1  # shifting index
				df = df.sort_index()  # sorting by index
	grouped_df =df.groupby(["problem", "solver"])
	grouped_df_mean = grouped_df.mean()
	pivot_table = grouped_df_mean.pivot_table(index='solver', columns='problem', values=value)
	n = pivot_table.shape[1]//max_n + 1
    # Plot each 'problem' in a separate subplot
	fig = plt.figure(figsize=(3 * max_n, 3 *n))
	colors = get_evenly_spaced_colors(len(pivot_table.index))
        
	for i, problem in enumerate(pivot_table.columns):
		ax = plt.subplot(n, max_n, i+1)
		pivot_table.plot.bar(y=problem, use_index = True, color = colors, ax = ax,)
		ax.set_title(f'Problem = {problem}')
		ax.set_xticks([]) 
        
        # Hide the legend on individual subplots
		ax.get_legend().remove()
    
	legend_labels = [f'{solver}' for solver in pivot_table.index]

	legend_handles = [Line2D([0], [0], color=color, linewidth=4) for color in colors]
    
	fig.legend(legend_handles, legend_labels, title='Legend',
                              bbox_to_anchor=(1, 0), loc='lower right')

	plt.tight_layout()
	plt.show()

## utils function for handling tsplib95 problems

def find_number_of_nodes(input_string :  str):
	# Use a regular expression to find the number at the end of the string
	match = re.search(r'\d+$', input_string.removesuffix(".tsp"))

	# If a match is found, extract and return the number
	if match:
		suffix_number = int(match.group())
		return suffix_number
	else:
		return inf

def get_tsp_files_available(folder, limit_number_of_files= None):

	# List all files in the folder
	all_files = os.listdir(folder)

	# Filter files with the specified extension
	filtered_files = [file for file in all_files if file.endswith(".tsp")]

	filtered_files.sort(key = find_number_of_nodes)
	if limit_number_of_files :
		filtered_files = filtered_files[:limit_number_of_files]
	return filtered_files


def get_positions(problem):
	coords = problem.node_coords.values()
	display_data = problem.display_data.values()
	if coords :
		return np.array(list(coords))
	elif display_data:
		return np.array(list(display_data))
	else :
		dima = get_distance_matrix(problem)
		mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
		return np.array(mds.fit_transform(np.asarray(dima)))


def get_distance_matrix(problem):
	G = problem.get_graph()
	return nx.to_numpy_array(G)


def check_best_solution(name):
	try:
		sol = tsplib95.load_solution("TSP_Instances/opt.tour/" + name.removesuffix(".tsp") + ".opt.tour")
		problem = tsplib95.load("TSP_Instances/" + name.removesuffix(".tsp") + ".tsp" )
		return [i-1 for i in sol.tours[0]], problem.trace_tours(sol.tours)[0]
	except FileNotFoundError:
		return [], inf




def find_solution_one_solver(problem :  tsplib95.models.Problem, solver, verbose= True, solution_check= inf):

	# now solve problem
	status, solution, path_length = solver(problem, verbose = verbose)
	if verbose :
		match status:
			case "OPTIMAL":
				log.info("Solution found was proven to be optimal")
			case "FEASIBLE" :
				log.info("Solution found was not proven to be optimal but a solution was found nonetheless")
			case "INFEASIBLE":
				log.info("The problem was proven to be infeasible ")

	if solution_check < path_length and status == "OPTIMAL":
		log.warning(f"solution found is not optimal, optimal path length is {solution_check} but solution found is {path_length} ")

	return status, solution, path_length


def find_solution(problem, solvers, solution_check= True, folder = "TSP_instances/", solution_save_file = "save.csv", verbose = True):
	try : 
		with open(solution_save_file, "r"):
			pass
	except FileNotFoundError :
		with open(solution_save_file, "a") as f:
			f.write("model,problem,status,len,time")
	if isinstance(problem, str):
		# load tsp instance
		problem = tsplib95.load(folder + problem)

	if not isinstance(problem, tsplib95.models.Problem):
		raise TypeError(f" problem must be either a tsplib95 problem or a string but found type : {type(problem)}")

	if verbose:
		log.info(problem.name)

	if not is_iterable(solvers):
		solvers = { "ours" : solvers}
	
	statuses, solutions, paths_length, times = {}, {}, {}, {}
	if solution_check :
		solutions["file solution"], paths_length["file solution"] = check_best_solution(problem.name)
		times["file solution"] = None
	for name,solver in solvers.items():
		
		if verbose:
			log.info(name)

		start = time.time()
		statuses[name], solutions[name], paths_length[name] = find_solution_one_solver(problem, solver, False, paths_length["file solution"] if solution_check else inf)
		times[name] = time.time() - start

		if verbose:
			log.info(name + " DONE ")

		with open(solution_save_file, "a") as f:
			f.write(",".join([name, problem.name, statuses[name], str(paths_length[name]), str(times[name]) ]) + "\n") # type: ignore	
	if verbose:
		print_solution(problem, solutions, lambda name : f"{name}(len : {paths_length[name]}, time : {times[name]})")

	
	return statuses, solutions, paths_length