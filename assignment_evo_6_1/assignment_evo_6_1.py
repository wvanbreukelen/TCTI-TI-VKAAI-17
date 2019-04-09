import random
from functools import reduce


class EvolutionaryAlgorithm:
    def __init__(self, populationSize):
        """ Initialize a new instance of a Evolutionary Algorithm.

        Arguments:
            populationSize {int} -- Size of population.
        """

        self.population = [self.GenerateChromosome()
                           for i in range(populationSize)]

    def GenerateChromosome(self):
        """ Generate a new chromosome. The chromosome contains ten unique shuffled numbers between 0 and 10.

        Returns:
            list -- New chromosome.
        """

        cards = list(range(1, 11))
        random.shuffle(cards)
        return cards

    def SplitChromosome(self, chromosome):
        """ Split a chromosome in two.

        Arguments:
            chromosome {list} -- Chromosome representation.

        Returns:
            list, list -- Split representation within two lists.
        """

        halfsize = int(len(chromosome)/2)

        pileA = chromosome[:halfsize]
        pileB = chromosome[halfsize:]

        return pileA, pileB

    def CalculateFitness(self, chromosome):
        """ Calculate the fitness of one chromosome.

        Arguments:
            chromosome {list} -- Chromosome representation.

        Returns:
            int -- Fitness of chromosome.
        """

        pileOne, pileTwo = self.SplitChromosome(chromosome)

        sumPileOne = sum(pileOne)
        sumPileTwo = reduce(lambda x, y: x * y, pileTwo)

        return abs(36 - sumPileOne) + abs(360 - sumPileTwo)

    def CalculatePopulationFitness(self):
        return [(self.CalculateFitness(chromosome), chromosome) for chromosome in self.population]

    def CreateChildChromosome(self, parentA, parentB, mutationChance):
        """ Create a new child chromosome based on two parents.

        Arguments:
            parentA {list} -- Parent chromosome A.
            parentB {list} -- Parent chromosome B.
            mutationChance {float} -- Chance of mutution. Must be between 0.0 and 1.0

        Returns:
            list -- New child chromosome.
        """

        child = self.Crossover(parentA.copy(), parentB.copy())

        if random.random() > mutationChance:
            child = self.Mutate(child)

        return child

    def Crossover(self, parentA, parentB):
        """ Perform single-point crossover over two chromosomes. Slice point is the exact middle of the chromosome.

        Arguments:
            parentA {list} -- Chromosome one.
            parentB {list} -- Chromesome two.

        Returns:
            list -- New generated chromosome.
        """

        sliceIndex = int(len(parentA) / 2)

        partA = parentA[:sliceIndex]
        partB = parentB[sliceIndex:]

        return partA + partB

    def Mutate(self, chromosome):
        """ Mutate a chromosome.

        Arguments:
            chromosome {list} -- Chromosome representation.

        Raises:
            ValueError -- [description]

        Returns:
            list -- Mutated chromosome.
        """

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
            # Chromosome does not contain a mutating gene anymore, so throw an exception.
            raise ValueError("Invalid chromosome: {}".format(chromosome))

        return child

    def GenerateNewGeneration(self, retainPercentage, mutationChance, randomSelection):
        """ Generate an new population.

        Arguments:
            retainPercentage {float} -- Percentage of indiviuals to retain in the new population. Input must be between 0% and 100%
            mutationChance {float} -- Chance of mutation. Must be between 0.0 and 1.0
            randomSelection {float} -- Chance that an indiviual is retained. Must be between 0.0 and 1.0

        Returns:
            list -- New population, also stored within self.population
        """

        newPopulation = []

        # Sort our current population based on the fitness.
        sortedPopulation = sorted(self.CalculatePopulationFitness())

        # Calculate the amount of genes to retain based on the retain percentage.
        if retainPercentage == 0:
            amountToRetain = 0
        else:
            amountToRetain = int(len(sortedPopulation) /
                                 (100 / retainPercentage))

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
        """ Calculate the total fitness for the whole population.

        Returns:
            int -- The sum fitness.
        """

        sum = 0
        for chromosome in self.CalculatePopulationFitness():
            sum += chromosome[0]

        return sum

    def GetBestOfPopulation(self):
        """ Return the sorted representation of all indiviuals based on their fitness.

        Returns:
            list -- Sorted population
        """

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
