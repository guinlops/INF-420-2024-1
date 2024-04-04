/* REFERENCIAS:
https://www.youtube.com/watch?v=dvWk0vgHijs&ab_channel=MaheshHuddar
https://www.geeksforgeeks.org/a-search-algorithm/
https://blog.goodaudience.com/solving-8-puzzle-using-a-algorithm-7b509c331288
https://github.com/Subangkar/N-Puzzle-Problem-CPP-Implementation-using-A-Star-Search
 */


#include <iostream>
#include <vector>
#include <algorithm>
#define N 3 //Numero de linhas e colunas
using namespace std;
int n_passos=1;

class Tab {
public:
  int tabuleiro[N][N], g, f;
  Tab* parent;
  Tab () {
    g = 0;
    f = 0;
    parent = NULL;
  }
  bool operator == (Tab a) {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        if (this->tabuleiro[i][j] != a.tabuleiro[i][j])
          return false;
    return true;
  }

  void print () {
    cout<<"Passo "<<n_passos<<"\n";
    cout << "g = " << g << " e f = " << f << endl;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++)
        cout << tabuleiro[i][j] << " ";
      cout << endl;
    }
    std::cout<<"\n";
    n_passos++;
  }
};


int heuristica (Tab from, Tab to) {
    int ret = 0;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        if (from.tabuleiro[i][j] != to.tabuleiro[i][j])
          ret++;
    return ret;
}


bool isinset (Tab a, vector <Tab> b) {
  for (int i = 0; i < b.size(); i++)
    if (a == b[i])
      return true;
  return false;
}

void addTabs (Tab current, Tab Tabfinal, int newi, int newj, int posi, int posj, vector <Tab>& openset, vector <Tab> closedset) {
  Tab newstate = current;
  swap (newstate.tabuleiro[newi][newj], newstate.tabuleiro[posi][posj]); //Faz a troca de posicoes;



  if (!isinset(newstate, closedset) && !isinset(newstate, openset)) { //Se não esta nem no closed nem no open set

      newstate.g = current.g + 1; //DEPTH do novo nodo se torna o do atual +1 
      newstate.f = newstate.g + heuristica(newstate, Tabfinal);
      Tab* temp = new Tab();
      *temp = current;
      newstate.parent = temp;
      openset.push_back(newstate); //Adiciona o a nova Tab ao newState
  }
}

void adjacentes (Tab tab_Atual, Tab Tabfinal, vector <Tab>& openset, vector <Tab>& closedset) {
  int i, j, posi ,posj;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (tab_Atual.tabuleiro[i][j] == 0) { //Procura o "espaco" no tabuleiro
        posi = i;
        posj = j;
        break;
      }
  i = posi;
  j = posj;



    //A partir do espaco em branco verifico no tabuleiro suas posicoes 4 possíveis posicoes adjacentes:
  if (i - 1 >= 0) //Movimento da esquerda para direita
    addTabs(tab_Atual, Tabfinal, i - 1, j, posi, posj, openset, closedset);
  if (i + 1 < N)
    addTabs(tab_Atual, Tabfinal, i + 1, j, posi, posj, openset, closedset);
  if (j + 1 < N)
    addTabs(tab_Atual, Tabfinal, i, j + 1, posi, posj, openset, closedset);
  if (j - 1 >= 0)
    addTabs(tab_Atual, Tabfinal, i, j - 1, posi, posj, openset, closedset);
}

bool passos(Tab tab_Atual, vector<Tab> &v_passos) {
    Tab *temp = &tab_Atual;
    while(temp != NULL) {
        v_passos.push_back(*temp);
        temp = temp->parent;
    }
    return true;
}


bool comp_tabs (Tab a, Tab b) {
  return a.f < b.f;
}

vector<Tab> v_passos; //Vetor de passos até a resoluçãop

bool Astar (Tab inicial,Tab final) {
  vector <Tab> openset; //cria vetor de Tabuleiros a serem processados  
  vector <Tab> closedset; //Cria o vetor de Tabuleiros já processados



  //Seta parametros do tabuleiro incial;
  Tab current;
  inicial.g = 0;
  inicial.f = inicial.g + heuristica(inicial, final);
  std::cout<<"Comp1 : inicial.f: "<<inicial.f<<"\n";


  openset.push_back(inicial);

  while (!openset.empty()) {

    sort(openset.begin(), openset.end(), comp_tabs);
    current = openset[0];
    if (current == final){
       return passos(current, v_passos);}
    openset.erase(openset.begin());
    closedset.push_back(current);
    adjacentes(current, final, openset, closedset);
  }
  return false;
}

int main () {
 Tab start, final;
 //g++ main.cpp -o main
 //main<in.txt> out.txt

  for (int i = 0; i < N; i++){
     for (int j = 0; j < N; j++){
         cin >> start.tabuleiro[i][j];}}
  
  for (int i = 0; i < N; i++){ 
    for (int j = 0; j < N; j++){
         cin >> final.tabuleiro[i][j];}}


  if (Astar(start, final)) {
    for (int i = v_passos.size() - 1; i >= 0; i--)
      v_passos[i].print();
  }
  else cout << "Não tem solução" << endl;
  return 0;
}