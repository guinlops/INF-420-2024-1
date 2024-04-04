# Nome:
# Matricula: 

from Posicao import Posicao
from AEstrela import AEstrela
from QuebraCabeca import QuebraCabeca
from QuebraCabecaImp import QuebraCabecaImp


class AEstrelaImp(AEstrela):
    
    def __init__(self) :
        self.solucao=[]
        self.tab_sol = [[1, 2, 3], [4, QuebraCabecaImp.VAZIO, 5], [6, 7, 8]]
        self.g=0


    
   

    def get_f(self,h,g):
        return h+g
    

    def fazTroca(self,tab:QuebraCabeca, i,j,x,y): ##tab é passado como referencia???? assumindo que nao
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
            self.temp.setTab(tab.getTab())### temp recebe o tabuleio atual]

            ##self.i," ",self.j," -> vazio") qual i e qual j vao ser trocadas pelas x e y da posicao vazia
            self.i=self.listaMovPossiveis[i].getLinha() 
            self.j=self.listaMovPossiveis[i].getColuna() 

            ##Define x e y como sendo as posicoes do vazio
            self.x=self.temp.getPosVazio().getLinha()
            self.y=self.temp.getPosVazio().getColuna()

            ##print(self.i," ",self.j," -> ",self.x," ", self.y)

            ##Gera um vetor com uma troca possivel
            self.temp.setTab(self.fazTroca(self.temp,self.i,self.j,self.x,self.y))
           
            ##Verifica se o hash dessa tab esta em closed set, ou seja, verifica se esse tab ja nao foi "encontrado"
            if( not(self.isInSet(self.temp,self.closedSet)) ):
                self.temp.setf(self.g)
                ##openSet.add(self.temp)
                openSet.append(self.temp)
                
           
            
            
        
        
        
       
        
        

        ##print("-------------------------------------------------------")
        """  for i in range(len(openSet)):
            print("hash: ",openSet[i].hashCode())
            print("h: ",openSet[i].getValor())
            print("f: ",openSet[i].getf())
            print(openSet[i].toString())
            """
    





    def Astar(self, tab_inicial:QuebraCabeca,closedSet):
        print("-------------------------------------------------------")
        self.openSet=[] ##open set é um set de objetos
        print("g (depth): ", self.g)
        print("h: ",tab_inicial.getValor())
        print("f: ",tab_inicial.getf())
        print(tab_inicial.toString())
         ##Se tiver ordenado, return
        if(tab_inicial.isOrdenado()):
            
            tab_inicial.toString()
            print("ordenado!")
            print("-------------------------------------------------------")
            return True
        
        #Adiciona o hash da tab atual ao closedSet      
        closedSet.add(tab_inicial.hashCode())
  
        #Gera Possiveis movimentos a partir da tab inicial e os coloca no openSet
        self.geraTabs(tab_inicial,self.openSet, closedSet)

        #o OpenSet é ordenado com base na heuristica
        self.openSet=sorted(self.openSet, key=lambda obj: obj.getf()) ## ordenacao com base na heuristica
        
        """  print("Opcoes possíveis de forma ordenda: ")
        for i in range(len(self.openSet)):
            print("hash: ",self.openSet[i].hashCode())
            print("g: ", self.g)
            print("h: ",self.openSet[i].getValor())
            print("f: ",self.openSet[i].getf())
            print(self.openSet[i].toString()) """
        ##Ordenar o set com base em sua heuristica e pegar o primeiro objeto



    ##RESOLVER O OPENSET, FAZER COM QUE ELE VISITE OUTROS ESTADOS ANTERIORES, JA QUE UM CAMINHO SÓ PODE NAO SER O IDEAL, ELE PODERIA VOLTAR A OUTROS. USANDO UM HEAP TVZ?

       
        
        print("Nao ta ordenado!!")
        tab_inicial.toString()

        newTab=QuebraCabecaImp()
        ##Assumindo que o deslocamento é de : a posicao vazia swap posicao passada para o vetor de solucao:
        if(len(self.openSet)!=0):
            self.solucao.append(Posicao(self.openSet[0].getPosVazio().getLinha(),self.openSet[0].getPosVazio().getColuna()))
            newTab=self.openSet[0]
      
       
        self.openSet=[]
        print("-------------------------------------------------------")
        self.Astar(newTab,closedSet)
        


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
 
        # Count inversions in given 8 puzzle
        inv_count = self.getInvCount(tab)
    
        # return true if inversion count is even.
        return (inv_count % 2 == 0)



    def getSolucao(self, qc:QuebraCabeca):


        #print("eh solucionavel?: ", self.isSolvable(qc.getTab()))

        if(not self.isSolvable(qc.getTab())):
            print("Nao eh solucionavel")
        else:
            print("Eh solucionavel")
            self.closedSet=set() ## closed set é um set de hashs
            self.Astar(qc,self.closedSet)
       
        
        return self.solucao

