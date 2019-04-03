import random
from functools import reduce


class EvolutionaryAlgorithm:
    def __init__(self, populationSize):
        self.population = [self.createChromosome()
                           for i in range(populationSize)]

    def createChromosome(self):
        # single chromosome containing all starting values of cards
        cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        random.shuffle(cards)
        return cards

    def splitChromosome(self, chromosome):
        halfsize = int(len(chromosome)/2)

        pile_0 = chromosome[:halfsize]
        pile_1 = chromosome[halfsize:]

        return pile_0, pile_1

    def calculateFitness(self, chromosome):
        pile_0, pile_1 = self.splitChromosome(chromosome)

        sumPileZero = sum(pile_0)
        sumPileOne = reduce(lambda x, y: x*y, pile_1)

        return abs(36-sumPileZero)+abs(360-sumPileOne)

    def calculatePopulationFitness(self):
        return [(self.calculateFitness(chromosome), chromosome) for chromosome in self.population]

    def createChildChromosome(self, parentA, parentB, splitAmount):
        child = []
        splitIndexes = []
        for i in range(splitAmount):
            splitIndexes.append(random.randint(0, len(parentA)))

        return child

    def performCrossover(self, amountOfCrossovers):
        crossoverIndexes = []

        for x in range(amountOfCrossovers):
            # Select the crossover index.
            crossoverIndex = random.randint(0, 9)

            if crossoverIndex not in crossoverIndexes:
                crossoverIndexes.append(crossoverIndex)

    def performMutation(self):
        pass

    def generateNewGeneration(self):
        pass


def main():
    EA = EvolutionaryAlgorithm(20)
    EA.calculatePopulationFitness()
    EA. generateChildren()


if __name__ == "__main__":
    main()
