import numpy as np
import re
import string_utils
import random
import operator
import time


POPULATION_SIZE = 500
ALPHABET_SIZE = 26
PROBABILITY_OF_MUTATION = 0.30
PERCENTAGE_OF_ELITISM = 0.15
PROBABILITY_OF_CROSS_OVER = 0.65
NUMBER_OF_CROSS_OVER_POINTS = 5

MAX_FITNESS = 0
class Decoder():
    def __init__(self, encodedText):
        self.encodedTxt = encodedText
        self.cleanEncoded = set(re.split('[^a-zA-Z]', self.encodedTxt))
        self.uniqueEncodedWordCount = len(self.cleanEncoded)
        self.key = None
        self.chars = 'abcdefghijklmnopqrstuvwxyz'
        self.alphabetSize = len(self.chars)
        self.chromosomeSize = 26
        self.alphabetNumbers = [i for i in range(self.alphabetSize)]
        self.mutationProbability = PROBABILITY_OF_MUTATION
        self.allGenerations = []
        self.generationMaxFitnesses = []
        
        
    def readInputTxt(self):
        f = open('global_text.txt','r')
        self.rawGlobalTxt = (f.read()).lower()
    
    def cleanTxt(self, text):
        words = re.split('[^a-zA-Z]', text)
        words = set(words)
        return words
    
    def createFirstGeneration(self, n = POPULATION_SIZE):
        self.populationSize = n
        firstGeneration = set()
        i = 0
        while (i != n):
            newChromosome = string_utils.shuffle(self.chars)
            if(newChromosome not in firstGeneration):
                i += 1
                firstGeneration.add(newChromosome)
        self.population = list(firstGeneration)
        self.allGenerations = [firstGeneration]
        
    def decrypt(self, chromosome):
        encodedTxt = self.encodedTxt.lower()
        for i in range(ALPHABET_SIZE):
            encodedTxt = encodedTxt.replace(chromosome[i], self.chars[i].upper())
        decryptedTxt = encodedTxt.lower()
        return decryptedTxt
    
    def calcFitness(self, chromosome):
        decryptedTxt = self.decrypt(chromosome)
        cleanDecrypt = self.cleanTxt(decryptedTxt)
        
        fitness = len(cleanDecrypt & self.globalWords)
        global MAX_FITNESS
        if(fitness > MAX_FITNESS):
            MAX_FITNESS = fitness
        return fitness

    def sortGeneration(self):
        chrom_Fitness = dict()
        for chromosome in self.population:
            chrom_Fitness[chromosome] = self.calcFitness(chromosome)
        sortedPopulation = dict(sorted(chrom_Fitness.items(), key=operator.itemgetter(1),reverse=True))
        self.sortedPopulation = sortedPopulation
        self.generationMaxFitnesses.append(MAX_FITNESS)
        
    def isKeyFound(self, chromosome):
        decryptedTxt = self.decrypt(chromosome)
        cleanDecrypt = self.cleanTxt(decryptedTxt)
        if(len(cleanDecrypt & self.globalWords) == len(cleanDecrypt)):
            return True
        return False

    def getFitnessOfGeneration(self):
        n = 0
        fitness = 0
        for chromosome in self.population:
            n += 1
            fitness += self.calcFitness(chromosome)
        return fitness/n
    
    def selectParents(self, n = 500, elitismPercentage = PERCENTAGE_OF_ELITISM ):
        sortedChromosomes = list(self.sortedPopulation.keys())
        probabilities = list(self.sortedPopulation.values())
        elitesNumber = int(n*elitismPercentage)
        probabilities = list(np.divide(probabilities, sum(probabilities)))
        
        parents = list(np.random.choice(sortedChromosomes, n - elitesNumber,
              p=probabilities))
        return parents

    def oneChild(self, mother, father, 
            crossoverPoints = NUMBER_OF_CROSS_OVER_POINTS, Pc = PROBABILITY_OF_CROSS_OVER):
        child = list(mother)
        dad = set(father)
        fromMomChars = set()
        fromMom = set(random.sample(self.alphabetNumbers, crossoverPoints))
        for i in fromMom:
            fromMomChars.add(child[i])
        dadTemp = list(dad - fromMomChars)
        for i in range(self.chromosomeSize):
            if(i in fromMom):
                continue
            if(dadTemp):
                child[i] = dadTemp[0]
                dadTemp.pop(0)
        return ''.join(child)
        
    def positionBasedCrossover(self, mother, father, 
            crossoverPoints = NUMBER_OF_CROSS_OVER_POINTS, Pc = PROBABILITY_OF_CROSS_OVER):
        if (mother == father or random.random() > Pc):
            return mother, father
        child1 = self.oneChild(mother, father, crossoverPoints, Pc)
        child2 = self.oneChild(father, mother, crossoverPoints, Pc)
        return child1, child2

    def mutate(self, chromosome, mutationProbability = PROBABILITY_OF_MUTATION):
        if(random.random() < mutationProbability):
            chromosomeList = list(chromosome)
            swappingParticles = random.sample(self.alphabetNumbers, 2)
            temp = chromosomeList[swappingParticles[1]]
            chromosomeList[swappingParticles[1]] = chromosomeList[swappingParticles[0]]
            chromosomeList[swappingParticles[0]] = temp
            return ''.join(chromosomeList)
        return chromosome

    def createNewPopulation(self, parentsCount = POPULATION_SIZE, elitismPercentage = PERCENTAGE_OF_ELITISM,
            crossoverPoints = NUMBER_OF_CROSS_OVER_POINTS, Pc = PROBABILITY_OF_CROSS_OVER, 
            mutationProbability = PROBABILITY_OF_MUTATION):
        sortedChromosomes = list(self.sortedPopulation.keys())
        parents = self.selectParents(parentsCount, elitismPercentage) #liste
        newPopulation = sortedChromosomes[0: int(elitismPercentage*parentsCount)]
        parentsCount = len(parents)
        for i in range(0,parentsCount-1,2):
            child1,child2 = self.positionBasedCrossover(parents[i], parents[i+1])
            child1 = self.mutate(child1, mutationProbability)
            child2 = self.mutate(child2, mutationProbability)
            newPopulation.append(child1)
            newPopulation.append(child2)
        self.population = newPopulation
        self.allGenerations.append(newPopulation)

    def decodeWithKey(self,key):
        decoded = list(self.decrypt(key))
        encoded = list(self.encodedTxt)
        for i in range(len(encoded)):
            if encoded[i].isupper():
                decoded[i] = decoded[i].upper()
        return ''.join(decoded)


        
    def isInLocalMax(self, n, currentFitness, desiredFitness):
        if(currentFitness/desiredFitness > 0.9):
            return False
        if(len(self.generationMaxFitnesses) > n + 1):           
            if(len(set(self.generationMaxFitnesses[-n:]) ) == 1):
                return True
        return False
    
    def resetGeneration(self):
        self.createFirstGeneration(500)
        self.sortGeneration()
        self.generationMaxFitnesses = []
        global MAX_FITNESS 
        MAX_FITNESS = 0
    
        
    def decode(self):
        self.readInputTxt()
        self.globalWords = self.cleanTxt(self.rawGlobalTxt)
        numberOfUniqueWords = len(self.cleanEncoded)
        self.createFirstGeneration()
        self.sortGeneration()
        keyFound = self.isKeyFound(next(iter(self.sortedPopulation.keys())))
        i = 0
        
        while not keyFound:
            i+=1
            self.createNewPopulation()
            self.sortGeneration()
            keyFound = self.isKeyFound(next(iter(self.sortedPopulation.keys())))
            if(self.isInLocalMax(160, MAX_FITNESS, numberOfUniqueWords)):
                self.resetGeneration()

        self.decodedTxt = self.decodeWithKey(next(iter(self.sortedPopulation.keys())))
        return self.decodedTxt
            
        

f = open('encoded_text.txt','r')
encodedTxt = f.read()
d = Decoder(encodedTxt)
start = time.time()
decoded = d.decode()
end = time.time()