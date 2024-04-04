# Nome: Guilherme Nunes Lopes   
# Matricula: 105462

from Posicao import Posicao
from AEstrela import AEstrela
from QuebraCabeca_modified import QuebraCabeca
from QuebraCabecaImp_modified import QuebraCabecaImp

import sys
sys.setrecursionlimit(10000) ##Evita RecursionError: maximum recursion depth exceeded para que minha solucao possa ser implementada de forma recursiva
class AEstrelaImp(AEstrela):
    
    def __init__(self) :
        self.solucao=[]
        self.tab_sol = [[1, 2, 3], [4, QuebraCabecaImp.VAZIO, 5], [6, 7, 8]]
        self.g=0


    
   

    def get_f(self,h,g):
        return h+g
    

    def fazTroca(self,tab:QuebraCabeca, i,j,x,y): ##Faz a troca de posicoes e retorna um tabuleiro int[][]
        self.aux=tab.getTab()
        self.aux[i][j],self.aux[x][y]=self.aux[x][y],self.aux[i][j]
        tab=tab.setTab(self.aux)
        return self.aux



    def isInSet(self, tab:QuebraCabeca, closedSet):
        if tab.hashCode() in closedSet:
            return True
        
        return False

    def geraTabs(self, tab:QuebraCabeca, openSet, closedSet):
        self.listaMovPossiveis=tab.getMovePossiveis()
        self.g+=1
        ##print(len(self.listaMovPossiveis))
        ##for i in range(len(self.listaMovPossiveis)):
        ##print(self.listaMovPossiveis[i].getLinha()," ",self.listaMovPossiveis[i].getColuna())



        for i in range(len(self.listaMovPossiveis)):
            self.temp=QuebraCabecaImp()
            self.temp.setTab(tab.getTab())### temp recebe o tabuleiro atual

            ##self.i," ",self.j," -> vazio") qual i e qual j vao ser trocadas pelas x e y da posicao vazia
            self.i=self.listaMovPossiveis[i].getLinha() 
            self.j=self.listaMovPossiveis[i].getColuna() 

            ##Define x e y como sendo as posicoes do vazio
            self.x=self.temp.getPosVazio().getLinha()
            self.y=self.temp.getPosVazio().getColuna()

            ##print(self.i," ",self.j," -> ",self.x," ", self.y)

            ##Gera um tabuleiro a partir de troca possível de ser feita
            self.temp.setTab(self.fazTroca(self.temp,self.i,self.j,self.x,self.y))
           
            ##Verifica se o hash dessa tab esta em closed set, ou seja, verifica se esse tab ja nao foi "encontrado"
            if( not(self.isInSet(self.temp,self.closedSet)) ):
                self.temp.setf(self.g)
                #Adiciona esse tabuleiro a um vetor que contem todos os tabuleiros possíveis a partir o tabuleiro inicial
                openSet.append(self.temp)
         
            ##print("Informacoes para depuracao")
            """  for i in range(len(openSet)):
            print("hash: ",openSet[i].hashCode())
            print("h: ",openSet[i].getValor())
            print("f: ",openSet[i].getf())
            print(openSet[i].toString())
            """
    

    def Astar(self, tab_inicial:QuebraCabeca,closedSet):
       
        self.openSet=[] ##open set é um set de objetos

        #Informacoes de depuracao para cada tabuleiro percorrido.
        """  print("g (depth): ", self.g)
        print("h: ",tab_inicial.getValor())
        print("f: ",tab_inicial.getf())
        print(tab_inicial.toString()) """
         ##Se tiver ordenado, return
        if(tab_inicial.isOrdenado()):
            
            
            print("ordenado!")
            print(tab_inicial.toString())
            print("-------------------------------------------------------")
            return True
        
        #Adiciona o hash da tab atual ao closedSet      
        closedSet.add(tab_inicial.hashCode())
  
        #Gera Possiveis movimentos a partir da tab inicial e os coloca no openSet
        self.geraTabs(tab_inicial,self.openSet, closedSet)

        #o OpenSet é ordenado com base na heuristica
        self.openSet=sorted(self.openSet, key=lambda obj: obj.getf()) ## ordenacao com base na heuristica
        
        """  print("Informarcoes heuristicas para depuracao de possiveis tabuleiros ")
        for i in range(len(self.openSet)):
            print("hash: ",self.openSet[i].hashCode())
            print("g: ", self.g)
            print("h: ",self.openSet[i].getValor())
            print("f: ",self.openSet[i].getf())
            print(self.openSet[i].toString()) """
    
       
        ##print("Nao ta ordenado nesse ponto!!")
        ##tab_inicial.toString()

        #Cria um tab temporario
        newTab=QuebraCabecaImp()
        ##Assumindo que o deslocamento é de : a posicao vazia swap posicao passada para o vetor de solucao:
        if(len(self.openSet)!=0):
            self.solucao.append(Posicao(self.openSet[0].getPosVazio().getLinha(),self.openSet[0].getPosVazio().getColuna()))
            newTab=self.openSet[0] ##new tab vai ser o tabuleiro de "melhor" heuristica, ja que o vetor foi ordenado com base na heuristica
      
       
        self.openSet=[]#Vetor de possíveis tabuleiros é zerado, já que os possíveis vao ser relativos a newTab, e nao a tab_inicial.
        ##print("-------------------------------------------------------")
        self.Astar(newTab,closedSet)##Chama recursivamente a funcao Astar, até que o tabuleiro esteja ordenado. 
        


    def getInvCount(self,tab):
        inv_count = 0
        empty_value = -1
        n = 3  # Tamanho da matriz, assumindo que seja uma matriz 3x3

        # Convertendo a matriz em uma lista unidimensional
        flattened_tab = [element for row in tab for element in row]

        for i in range(n * n):
            for j in range(i + 1, n * n):
                if flattened_tab[j] != empty_value and flattened_tab[i] != empty_value and flattened_tab[i] > flattened_tab[j]:
                    inv_count += 1
        return inv_count
    

    def isSolvable(self,tab) :
 
        # Conta o numero de "Inversoes no tabuleiro"
        inv_count = self.getInvCount(tab)
    
        # return true se o numero de inversoes é par.
        return (inv_count % 2 == 0)



    def getSolucao(self, qc:QuebraCabeca):


        #print("eh solucionavel?: ", self.isSolvable(qc.getTab()))

        if(not self.isSolvable(qc.getTab())):
            print("Nao eh solucionavel")
        else:
            print("Eh solucionavel")
            self.closedSet=set() ## closed set é um set de hashs
            self.Astar(qc,self.closedSet) ##qc não é de fato modificado, mas sua cópia sim.

            
            
       
        
        return self.solucao

