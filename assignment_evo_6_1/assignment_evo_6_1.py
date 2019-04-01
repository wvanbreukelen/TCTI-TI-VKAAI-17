import random
from functools import reduce

class EvolutionaryAlgorithm:

    population = []


    def __init__(self, populationSize):
        self.population = [self.createChromosome() for i in range(populationSize)]

    def createChromosome(self):
        cards = [1,2,3,4,5,6,7,8,9,10] #single chromosome containing all starting values of cards
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
        sumPileOne = reduce(lambda x,y : x*y, pile_1)

        return abs(36-sumPileZero)+abs(360-sumPileOne)

    def calculatePopulationFitness(self):
        return [(self.calculateFitness(chromosome), chromosome) for chromosome in self.population]
    
    def createChildChromosome(self, parentA, parentB, mutationChance):

        child = self.crossover(parentA.copy(), parentB.copy())

        if random.randint(0,100) <= mutationChance:
            child = self.mutate(child)

        return child
    
    def crossover(self, parentA, parentB):

        splitIndex = int(len(parentA)/2)
        partA = parentA[:splitIndex]
        partB = parentB[splitIndex:]

        return partA+partB

    def mutate(self, chromosome):
        child = chromosome

        mutationIndex = random.randint(1, len(chromosome)-1)

        tempValue = child[mutationIndex]
        newValue = random.randint(1,10)
        while(newValue == tempValue):
            newValue = random.randint(1,10)

        if newValue in child:
            toSwapIndex = child.index(newValue)
            child[mutationIndex] = newValue
            child[toSwapIndex] = tempValue

        return child
        

    def generateNewGeneration(self, retainPercentage, mutationChance):
        newPopulation = []
        
        sortedPopulation = sorted(self.calculatePopulationFitness()) 
        retainIndex = int(len(sortedPopulation)/(100/retainPercentage))

        parents = []

        for passedParentIndex in range(0,retainIndex):
            parents.append(sortedPopulation[passedParentIndex][1])

        childAmount = len(self.population)-retainIndex
        children = []

        for child in range(childAmount):

            #Only accept valid children into the new generation

            newChild = []
            childIsValid = False
            while not childIsValid:
                newChild = self.createChildChromosome(random.sample(parents,1)[0], random.sample(parents,1)[0], mutationChance)

                errorFound = False
                for value in range(1,11):
                    if (newChild.count(value) > 1):
                        errorFound = True
                
                if not errorFound:
                    childIsValid = True
            children.append(newChild)    
           

        newPopulation = parents+children
        self.population = newPopulation
        return self.population

    def getGenerationFitness(self):
        sum = 0
        for chromosome in self.calculatePopulationFitness():
            sum += chromosome[0]

        return sum

    def getBestOfPopulation(self):
        return sorted(self.calculatePopulationFitness())[0]


def main():
    attempts = 100
    generations = 100
    populationSize = 50
    bestPercentageOfPopulationToAge = 50
    mutationChancePercentage = 10

    for i in range(attempts):
        EA = EvolutionaryAlgorithm(populationSize)
        for j in range(generations):
            EA.generateNewGeneration(bestPercentageOfPopulationToAge, mutationChancePercentage)
        print("Attempt\t", i + 1,"\tTotal Fitness:\t", EA.getGenerationFitness(),"\tafter\t",j+1,"\tgenerations" )
        bestOfPop = EA.getBestOfPopulation()
        print("Best result:\t", bestOfPop[1], "\twith fitness:\t", bestOfPop[0], '\n')
        

if __name__ == "__main__":
    main()