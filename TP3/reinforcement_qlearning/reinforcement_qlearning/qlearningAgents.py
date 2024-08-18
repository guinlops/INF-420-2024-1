# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).




#Nome: Guilherme Nunes Lopes
#Matricula: 105462
#Resposta para o passo 3: NÃO É POSSÍVEL


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "**** YOUR CODE HERE ****"
        #util.Counter() é a estrutura de dados fornecida pela util.py
        self.Qvalues=util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "**** YOUR CODE HERE ****"
        return self.Qvalues.get((state, action), 0.0) # se a chave não estiver presente, ele retorna o valor padrão especificado, que é 0.0 neste caso.
        #util.raiseNotDefined()



    #A função computeValueFromQValues calcula o valor máximo do Q-value para um dado estado, considerando todas as ações legais disponíveis nesse estado.
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "**** YOUR CODE HERE ****"
        allActions = self.getLegalActions(state)

        if len(allActions) == 0 : #estados terminais onde o agente não pode mais tomar ações.
          return 0.0
        else :
          qvalues_list = [] #Cria uma lista para armazenar os valores Q para cada ação possível
          for action in allActions :
              qvalues_list.append(self.getQValue(state, action))
          
          return max(qvalues_list) #Retorna o Qvalor maximo
        
        
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "**** YOUR CODE HERE ****"
        allActions = self.getLegalActions(state)
        
        if len(allActions) == 0 :
            return None
        
        possibleActions = []
        maxQvalue = self.getValue(state)
        for action in allActions :
            if(self.getQValue(state,action))>=maxQvalue:
                possibleActions.append(action)
        return random.choice(possibleActions)
        
        
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        AllActions=self.getLegalActions(state)
        #action = QLearningAgent.getAction(self,state)
        #self.doAction(state,action)
        #return action
        if util.flipCoin(self.epsilon) :
          return random.choice(AllActions)
        else :
          return self.getPolicy(state)
        


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #newQValue representa a atualização do valor Q para um par estado-ação (s,a). Ou seja, o Q(s,a) da fórmula do Q learning!!
        #getValue representa a função de maximização, que é usada para estimar o valor máximo possível que pode ser obtido a partir de um determinado estado.
        sample=reward+self.discount*self.getValue(nextState)
        newQValue=(1-self.alpha)*self.getQValue(state,action) + self.alpha*sample
        self.Qvalues[(state, action)] = newQValue


        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
       


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        featureVector = self.featExtractor.getFeatures(state, action)
        Qvalue = 0
        #o agente usa uma função de aproximação para calcular os Q-values dinamicamente com base em uma representação compacta do estado.
        for feature in featureVector :
            Qvalue += featureVector[feature]*self.getWeight(feature)
        return Qvalue

        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "**** YOUR CODE HERE ****"
        allFeatures = self.featExtractor.getFeatures(state, action)
        #Representação Linear para o q-learning Aproximado
        difference = reward + (self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        for feature in allFeatures:
            self.weights[feature] = self.getWeight(feature) + (self.alpha * difference * allFeatures[feature])
       
        ###util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "**** YOUR CODE HERE ****"
            pass
