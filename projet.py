import argparse
import utils
from functools import partial
from cp_SAT import solve_OrTools
from fourmil import ACO
from TSP_OR import incompletOR
from ortools.sat.python import cp_model

if __name__ == "__main__":
    available_solvers = {
        "complete" : partial(solve_OrTools, enforce= False),
        "complete_sup" : partial(solve_OrTools, supplementary_constraint = True, enforce= False),
        "complete_enf" : solve_OrTools,
        "inc_aco" : ACO,
        "complete_strategy" : partial(solve_OrTools, strategies= [[cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE], [cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MAX_VALUE]]),
        "enforce_strategy" : partial(solve_OrTools, strategies= [[cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE], [cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MAX_VALUE]], force_strategy = True),
        "OR_incomplete" : incompletOR,
    }
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Command-line runner of our project SAT solver')
    group = parser.add_mutually_exclusive_group()

    # Add arguments for problems selection
    group.add_argument('-p', '--problems', nargs='+', help='Spécifiez les problèmes que vous souhaitez résoudre')
    group.add_argument('-n', '--number_of_problems', default=10, type=int, help='Spécifiez le nombre de problèmes que vous souhaitez résoudre (Ils seront sélectionné par nombre croissant de nœuds)')

    # Add an argument for solvers selection
    parser.add_argument('-s', '--solvers', nargs='+',choices= available_solvers.keys() , default='all', help='Spécifiez les solveurs que vous souhaitez utiliser')

    # Add an argument for solution check
    parser.add_argument('--no_solution_check', dest="solution_check", action='store_false', help="Supprimer la vérification des fichiers de solution dans la bibliothèque")

    # Add argument for visualising solutions
    parser.add_argument('--visualise_solution',action='store_true', help="Ajouter une visualisation de toutes les solutions pour chaque problème")

    # Add an argument for verbose
    parser.add_argument('-v', '--verbose', action='store_true')

    # Add an argument for file_saving
    parser.add_argument("--filesave_path", type= str, help="Spécifiez un fichier dans lequel toutes les données correspondant aux solutions seront écrites.")

    # Parse the command-line arguments, including the default help argument
    args = parser.parse_args()

    if args.number_of_problems :
        problems = utils.get_tsp_files_available("TSP_Instances",args.number_of_problems)
    else:
        problems = args.problems

    if args.solvers == "all":
        solvers = available_solvers
    else :
        solvers = {key: available_solvers[key] for key in args.solvers}

    print(args.visualise_solution)
    for problem in problems:
        statuses, solutions, paths_length = utils.find_solution(problem, solvers, verbose=args.verbose, solution_check= args.solution_check, visualise_solution = args.visualise_solution, solution_save_file= args.filesave_path)
        to_print = f" problem : {problem:15}"
        for solver in solvers :
            val = f"{paths_length[solver]}({statuses[solver]})"
            val += " "*(20 -len(val))
            to_print += f" {solver} : {val}"
        print(to_print)