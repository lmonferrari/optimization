# The main problem is to organize flights that minimize the waiting time and the costs
# Destination: Rome

import random
import pandas as pd

people = [('Lisbon', 'LIS'),
          ('Madrid', 'MAD'),
          ('Paris', 'CDG'),
          ('Dublin', 'DUB'),
          ('Brussels', 'BRU'),
          ('London', 'LHR')]

destination = 'FCO'  # Rome

df_flights = pd.read_csv('./data/flights.txt', names=['Origin', 'Destination', 'Departure', 'Arrival', 'Ticket_cost'])

flights = {}

for register in df_flights.itertuples():
    origin = register.Origin
    destination = register.Destination
    departure = register.Departure
    arrival = register.Arrival
    ticket_cost = register.Ticket_cost

    key = (origin, destination)
    flights.setdefault(key, [])
    flights[key].append([departure, arrival, int(ticket_cost)])


def print_schedule(schedule):
    flights_id = -1
    total_cost = 0
    for i in range(len(schedule) // 2):
        # outbound flight information
        flights_id += 1
        _name = people[i][0]
        _origin = people[i][1]  # we are iterating over 6 cities at people
        _departure, _arrival, _cost = flights[(_origin, destination)][schedule[flights_id]]  # getting flights using keys and filtering
        total_cost += _cost

        print(f'1 - Origin: {_origin} - Destination {destination} - Departure: {_departure:6} - '
              f'Arrival: {_arrival:5} - Ticket cost: {_cost}')

        # data about the returning flight
        flights_id += 1
        _departure, _arrival, _cost = flights[(destination, _origin)][schedule[flights_id]]
        total_cost += _cost

        print(f'2 - Origin: {destination} - Destination {_origin} - Departure: {_departure:6} - '
              f'Arrival: {_arrival:5} - Ticket cost: {_cost}')


def get_minutes(s_hour):
    h_hour, m_hour = map(int, s_hour.split(':'))
    h_hour *= 60
    return int(h_hour + m_hour)


def cost_function(solution):
    total_cost = 0
    first_departure = get_minutes('23:59')
    last_arrival = get_minutes('00:00')
    flights_id = -1

    for i in range(len(solution) // 2):
        _origin = people[i][1]

        flights_id += 1
        _departure, _arrival, _cost = flights[(_origin, destination)][solution[flights_id]]
        total_cost += _cost

        if last_arrival < get_minutes(_arrival):
            last_arrival = get_minutes(_arrival)

        flights_id += 1
        _departure, _arrival, _cost = flights[(destination, _origin)][solution[flights_id]]
        total_cost += _cost

        if first_departure > get_minutes(_departure):
            first_departure = get_minutes(_departure)

    total_wait = 0
    flights_id = -1

    for i in range(len(solution) // 2):
        _origin = people[i][1]

        flights_id += 1
        _, _arrival, _ = flights[(_origin, destination)][solution[flights_id]]

        total_wait += last_arrival - get_minutes(_arrival)

        flights_id += 1
        _departure, _, _ = flights[(_origin, destination)][solution[flights_id]]

        total_wait += get_minutes(_departure) - first_departure

    return total_cost + total_wait


def crossover(domain, solution1, solution2):
    int_rand = random.randint(1, len(domain) - 2)
    return solution1[:int_rand] + solution2[int_rand:]


def mutation(domain, step, solution, mutation_rate):
    int_rand = random.randint(0, len(domain) - 1)
    mutant = solution

    if random.random() < mutation_rate:
        if solution[int_rand] != domain[int_rand][0]:
            mutant = solution[:int_rand] + [solution[int_rand] - step] + solution[int_rand + 1:]
        else:
            if solution[int_rand] != domain[int_rand][1]:
                mutant = solution[:int_rand] + [solution[int_rand] + step] + solution[int_rand + 1:]

    return mutant


def ga(domain, cost_function, population_size=100, step=1, elitism=0.2, n_generations=50, mutation_rate=0.2):
    population = [[random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]  # initial population
                  for _ in range(population_size)]

    n_elism = int(elitism * population_size)

    for i in range(n_generations):
        costs = [(cost_function(individual), individual) for individual in population]
        costs.sort()
        sorted_individuals = [individual for (cost, individual) in costs]
        population = sorted_individuals[:n_elism]

        while len(population) < population_size:
            i1 = random.randint(0, n_elism)
            i2 = random.randint(0, int(population_size-1))
            new_individual = crossover(domain, sorted_individuals[i1], sorted_individuals[i2])
            new_mutant = mutation(domain, step, new_individual, mutation_rate)
            population.append(new_mutant)

    return costs[0][1]


print('Processing...')
global_domain = [(0, len(flights[key]) - 1) for key in flights.keys()]
ga_solution = ga(global_domain, cost_function, population_size=100, step=1, n_generations=150, mutation_rate=0.2)
print('Done')
print_schedule(ga_solution)
print(cost_function(ga_solution))
