import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
from utils.functions import cluster_pop




def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }

    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_cvrp_bus_data_with_clustering(dataset_size, vrp_size):
    CAPACITY = 45

    #before clustering
    pop_size=100
    
    pop_dist = [1975132/4827184,2361745/4827184, 448404/4827184, 36683/4827184, 5220/4827184]    
    pop_demand = np.random.choice([1,2,3,4,5],size=pop_size, replace=True, p=pop_dist) 

    # Node locations clustered 
    locs = []
    demands = []
    for i in range(dataset_size) :
        pop = np.random.uniform(size=(pop_size, 2))
        pop_demand = np.random.choice([1,2,3,4,5],size=pop_size, replace=True, p=pop_dist) 
        loc, _ , demand = cluster_pop(pop, vrp_size, pop_demand)
        locs.append(loc.tolist())
        demands.append(demand.tolist())
        if (i%100) ==0 :
            print(f'{i}th data generated')
    return list(zip(
        np.random.uniform(size=(dataset_size,2)).tolist(), # depot location
        locs, # node location (dataset_size, vrp_size,2 )
        demands, #demand (dataset_size, vrp_size)
        np.full(dataset_size,CAPACITY).tolist()
    ))







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    # parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='all',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"


    problem = 'cvrp_bus'
    seed = 1234
    name = 'validation'
    graph_size = 20
    datadir = os.path.join('data', problem)
    distribution=[None]
    dataset_size=10000
    os.makedirs(datadir, exist_ok=True)


    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
        problem,
        "",
        graph_size, name, seed))
    
    assert not os.path.isfile(check_extension(filename)), \
        "File already exists! Try running with -f option to overwrite."

    np.random.seed(seed)

    dataset = generate_cvrp_bus_data_with_clustering(dataset_size, graph_size)


    print(dataset[0])   

    save_dataset(dataset, filename)
