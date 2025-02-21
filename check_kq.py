import pickle

with open("best.pickle", "rb") as f:
    best_genome = pickle.load(f)

print(f"Genome ID: {best_genome.key}")
print(f"Fitness: {int(round(best_genome.fitness))}")  
