import random
from functools import reduce


class EvolutionaryAlgorithm:

    population = []

    def __init__(self, populationSize):
        self.population = [self.CreateChromosome()
                           for i in range(populationSize)]

    def CreateChromosome(self):
        cards = list(range(1, 11))
        random.shuffle(cards)
        return cards

    def SplitChromosome(self, chromosome):
        halfsize = int(len(chromosome)/2)

        pile_0 = chromosome[:halfsize]
        pile_1 = chromosome[halfsize:]

        return pile_0, pile_1

    def CalculateFitness(self, chromosome):
        pileOne, pileTwo = self.SplitChromosome(chromosome)

        sumPileOne = sum(pileOne)
        sumPileTwo = reduce(lambda x, y: x * y, pileTwo)

        return abs(36 - sumPileOne) + abs(360 - sumPileTwo)

    def CalculatePopulationFitness(self):
        return [(self.CalculateFitness(chromosome), chromosome) for chromosome in self.population]

    def CreateChildChromosome(self, parentA, parentB, mutationChance):
        child = self.Crossover(parentA.copy(), parentB.copy())

        if random.random() > mutationChance:
            child = self.Mutate(child)

        return child

    def Crossover(self, parentA, parentB):
        sliceIndex = int(len(parentA) / 2)

        partA = parentA[:sliceIndex]
        partB = parentB[sliceIndex:]

        return partA + partB

    def Mutate(self, chromosome):
        child = chromosome

        mutationIndex = random.randint(0, len(chromosome)-1)
        oldGeneValue = child[mutationIndex]

        # Calculate a new value for a gene within the chromosome. It may not be the same as the old gene value.
        newGeneValue = random.randint(1, 10)
        while(oldGeneValue == newGeneValue):
            newGeneValue = random.randint(1, 10)

        if newGeneValue in child:
            swapIndex = child.index(newGeneValue)
            child[mutationIndex] = newGeneValue
            child[swapIndex] = oldGeneValue
        else:
            pass
            # Chromosome does not contain a mutating gene anymore, so throw an exception.
            #raise ValueError("Invalid chromosome: {}".format(chromosome))

        return child

    def GenerateNewGeneration(self, retainPercentage, mutationChance, randomSelection):
        newPopulation = []

        # Sort our current population based on the fitness.
        sortedPopulation = sorted(self.CalculatePopulationFitness())

        # Calculate the amount of genes to retain based on the retain percentage.
        amountToRetain = int(len(sortedPopulation) / (100 / retainPercentage))

        parents = []
        children = []

        # Add all parents to retain.
        for parentIndex in range(amountToRetain):
            if random.random() > randomSelection:
                parents.append(sortedPopulation[parentIndex][1])

        # In the case no parents where selected, we manually add some.
        for parentIndex in range(random.randint(1, amountToRetain)):
            parents.append(sortedPopulation[parentIndex][1])

        # Amount of children within the new population.
        amountOfChildren = len(self.population) - len(parents)

        while len(children) < amountOfChildren:
            # Only accept valid children into the new generation
            newChild = []
            childIsValid = False

            while not childIsValid:
                male = random.randint(0, len(parents) - 1)
                female = random.randint(0, len(parents) - 1)

                newChild = self.CreateChildChromosome(
                    parents[female], parents[male], mutationChance)

                errorFound = False
                for value in range(1, 11):
                    if (newChild.count(value) > 1):
                        errorFound = True

                if not errorFound:
                    childIsValid = True

            children.append(newChild)

        self.population = parents + children

        return self.population

    def GetGenerationFitness(self):
        sum = 0
        for chromosome in self.CalculatePopulationFitness():
            sum += chromosome[0]

        return sum

    def GetBestOfPopulation(self):
        return sorted(self.CalculatePopulationFitness())[0]


def main():
    attempts = 20
    generations = 100
    # populationSize = 50
    populationSize = 100
    retainPercentage = 50
    mutationChancePercentage = 0.1
    selectionChancePercentage = 0.1

    for i in range(attempts):
        EA = EvolutionaryAlgorithm(populationSize)
        for j in range(generations):
            EA.GenerateNewGeneration(
                retainPercentage, mutationChancePercentage, selectionChancePercentage)
        print("Attempt\t", i + 1, "\tTotal Fitness:\t",
              EA.GetGenerationFitness(), "\tafter\t", j+1, "\tgenerations")
        bestOfPop = EA.GetBestOfPopulation()
        print("Best result:\t", bestOfPop[1],
              "\twith fitness:\t", bestOfPop[0], '\n')


if __name__ == "__main__":
    main()
