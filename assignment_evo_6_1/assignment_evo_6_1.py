"""
Evolutionary Algorithm (exercise 6.1: cards problem) implementation by Kevin Nijmeijer and Wiebe van Breukelen.

We've chosen to implement this exercise using an multipoint swap crossover operator as we are dealing with an ordening problem.
Mutation is performed by randomlity selecting genes within the chromosome.

As an bonus, Kevin also implemented the order-based crossover operator. In our tests, order-based crossover was faster but took more generations to
get to a stable result.
The swap crossover operator was slower per generation but took less generations to get to a stable result.
"""


import random
from functools import reduce
from enum import Enum


class CrossoverOperator(Enum):
    SWAP = 0
    ORDER = 1


class EvolutionaryAlgorithm:
    def __init__(self, populationSize: int):
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

    def SplitChromosome(self, chromosome: list):
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

    def CalculateFitness(self, chromosome: list):
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
        """Calculate all fitnesses for all chromosomes within the population.

        Returns:
            tuple -- Tuple containing the fitness and chromosome itself.
        """

        return ((self.CalculateFitness(chromosome), chromosome) for chromosome in self.population)

    def CreateChildChromosomes(self, parentA: list, parentB: list, mutationRate: float, crossoverOp=CrossoverOperator.SWAP):
        """ Create a new child chromosomes based on two parents.

        Arguments:
            parentA {list} -- Parent chromosome A.
            parentB {list} -- Parent chromosome B.
            mutationChance {float} -- Chance of mutution. Must be between 0.0 and 1.0

        Returns:
            list -- New child chromosome.
        """
        if crossoverOp is CrossoverOperator.SWAP:
            # Multiple children using Swap Crossover
            children = self.SwapCrossover(parentA.copy(), parentB.copy())
        elif crossoverOp is CrossoverOperator.ORDER:
            # Single Child using order 1 crossover
            children = self.OrderOneCrossover(parentA.copy(), parentB.copy())

        for child in children:
            if random.random() > mutationRate:
                child = self.Mutate(child)

        return children

    def SwapCrossover(self, parentA: list, parentB: list):
        """ Perform multipoint swap crossover operator over two chromosomes. Crossover positions are determined randomly.

        Arguments:
            parentA {list} -- Chromosome one.
            parentB {list} -- Chromesome two.

        Returns:
            list -- New generated chromosome.
        """
        childA = [None for i in range(len(parentA))]
        childB = [None for i in range(len(parentA))]
        parentACopy = parentA.copy()
        parentBCopy = parentB.copy()

        crossoverPositions = []

        # Determine unique crossover positions.
        while int(len(parentA) / 2) > len(crossoverPositions):
            index = random.randint(0, 9)

            if index not in crossoverPositions:
                crossoverPositions.append(index)

        # Perform swap crossover.
        for i in range(int(len(parentA) / 2)):
            childA[crossoverPositions[i]] = parentA[crossoverPositions[i]]
            childB[crossoverPositions[i]] = parentB[crossoverPositions[i]]
            parentACopy.remove(parentB[crossoverPositions[i]])
            parentBCopy.remove(parentA[crossoverPositions[i]])

        # Fill in existing gaps with their responding originating values from their parents.
        for i in range(int(len(parentA))):
            if childA[i] is None:
                childA[i] = parentBCopy.pop()

            if childB[i] is None:
                childB[i] = parentACopy.pop()

        return [childA, childB]

    def OrderOneCrossover(self, parentA: list, parentB: list):
        """ Crossover using Order 1 crossover based on http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/Order1CrossoverOperator.aspx

        Arguments:
            parentA {[list]} -- Copy of first parent chromosome.
            parentB {[list]} -- Copy of second parent chromosome.

        Returns:
            list -- Single recombined child chromosome.
        """

        halfLen = int(len(parentA)/2)

        child = [None] * 10
        droppedAlleles = []

        dropdownStartIndex = random.randint(0, halfLen)
        dropdownEndIndex = dropdownStartIndex + 5

        for genIndex in range(dropdownStartIndex, dropdownEndIndex):
            child[genIndex] = parentA[genIndex]
            droppedAlleles.append(parentA[genIndex])

        for gen in droppedAlleles:
            if gen in parentB:
                parentB.remove(gen)

        for genIndex in range(len(child)):
            if child[genIndex] == None:
                child[genIndex] = parentB[0]
                parentB.remove(parentB[0])

        return [child]

    def Mutate(self, chromosome: list):
        """ Mutate a chromosome randomly.

        Arguments:
            chromosome {list} -- Chromosome representation.

        Returns:
            list -- Mutated chromosome.
        """

        child = chromosome.copy()

        mutationIndex = random.randint(0, len(chromosome) - 1)
        oldGeneValue = child[mutationIndex]

        # Calculate a new value for a gene within the chromosome. It may not be the same as the old gene value.
        newGeneValue = random.randint(1, 10)
        while(oldGeneValue == newGeneValue):
            newGeneValue = random.randint(1, 10)

        swapIndex = child.index(newGeneValue)
        child[mutationIndex] = newGeneValue
        child[swapIndex] = oldGeneValue

        return child

    def GenerateNewGeneration(self, retainPercentage: float, randomSelection: float, mutationRate: float, crossoverOp=CrossoverOperator.SWAP):
        """ Generate an new population.

        Arguments:
            retainPercentage {float} -- Percentage of indiviuals to retain in the new population. Input must be between 0.0 and 1.0
            mutationChance {float} -- Chance of mutation. Must be between 0.0 and 1.0
            randomSelection {float} -- Chance that an indiviual is retained. Must be between 0.0 and 1.0
            crossoverOperator {CrossoverOperator} -- Selected crossover operator.

        Returns:
            list -- New population, also stored within self.population
        """

        # Sort our current population based on the fitness.
        sortedPopulation = sorted(self.CalculatePopulationFitness())

        # Calculate the amount of genes to retain based on the retain percentage.
        if retainPercentage <= 0.0:
            amountToRetain = 0
        else:
            amountToRetain = int(len(sortedPopulation) /
                                 (1 / retainPercentage))

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
            # Select two parents out of the old generation.
            parentA = random.randint(0, len(parents) - 1)
            parentB = random.randint(0, len(parents) - 1)

            # Perform crossover to generate new children.
            children += self.CreateChildChromosomes(
                parents[parentA], parents[parentB], mutationRate, crossoverOp)

        # Remove overflow of children
        for _i in range(abs(amountOfChildren - len(children))):
            children.pop()

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
    generations = 50
    populationSize = 100
    retainPercentage = 0.5
    selectionChancePercentage = 0.1
    mutationRate = 0.1

    for i in range(attempts):
        EA = EvolutionaryAlgorithm(populationSize)
        for j in range(generations):
            EA.GenerateNewGeneration(
                retainPercentage, selectionChancePercentage, mutationRate, CrossoverOperator.SWAP)
        print("Attempt\t", i + 1, "\tTotal Fitness:\t",
              EA.GetGenerationFitness(), "\tafter\t", j+1, "\tgenerations")
        bestOfPop = EA.GetBestOfPopulation()
        print("Best result:\t", bestOfPop[1],
              "\twith fitness:\t", bestOfPop[0], '\n')


if __name__ == "__main__":
    main()
