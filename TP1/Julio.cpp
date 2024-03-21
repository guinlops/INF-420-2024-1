#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

using namespace std;


static constexpr int ME = 1;
static constexpr int OPP = 0;
static constexpr int NONE = -1;

struct Tile {
    int x, y, scrap_amount, owner, units,recycler,can_build;
    
    double tilePoints; //Para cada Tile, é atribuída uma pontuacao com base em sua distancia em relacao a uma tile que EU possuo.

    bool can_spawn, in_range_of_recycler;
        ostream& dump(ostream& ioOut) const {
        ioOut << x << " " << y;
        return ioOut;
    }
};
ostream& operator<<(ostream& ioOut, const Tile& obj) { return obj.dump(ioOut); }

double distancia(const Tile &a,const Tile &b){
    return abs(a.x - b.x) + abs(a.y - b.y);
}


int main()
{
    int width;
    int height;
    cin >> width >> height; cin.ignore();

    // game loop
    while (1) {
        vector<Tile> tiles;
        vector<Tile> my_tiles;
        vector<Tile> opp_tiles;
        vector<Tile> neutral_tiles;
        vector<Tile> my_units;
        vector<Tile> opp_units;
        vector<Tile> my_recyclers;
        vector<Tile> opp_recyclers;

        tiles.reserve(width * height);

        int my_matter;
        int opp_matter;
        cin >> my_matter >> opp_matter; cin.ignore();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int scrap_amount;
                int owner; // 1 = me, 0 = foe, -1 = neutral
                int units;
                int recycler;
                int can_build;
                int can_spawn;
                int in_range_of_recycler;
                cin >> scrap_amount >> owner >> units >> recycler >> can_build >> can_spawn >> in_range_of_recycler; cin.ignore();

                Tile tile = {x, y, scrap_amount, owner, units, recycler == 1, can_build == 1, can_spawn == 1, in_range_of_recycler == 1};
                tiles.emplace_back(tile);

                //para cada tile, ele faz o reconhecimento das informacoes e faz um push_back em um vector.
                if (tile.owner == ME) { //Se o dono dessa tile sou eu:
                    my_tiles.emplace_back(tile);
                    if (tile.units > 0) {
                        my_units.emplace_back(tile);
                    } else if (tile.recycler) {
                        my_recyclers.emplace_back(tile);
                    }
                } else if (tile.owner == OPP) { //Senao
                    opp_tiles.emplace_back(tile);
                    if (tile.units > 0) {
                        opp_units.emplace_back(tile);
                    } else if (tile.recycler) {
                        opp_recyclers.emplace_back(tile);
                    }
                } else {
                    neutral_tiles.emplace_back(tile);
                }
            }
        }

        vector<string> actions; //Cria um vetor de actions;
        vector<Tile> targetTiles; //Cria um vetor que guarda todas as possíveis tiles a serem visitadas por cada agente.
        vector<Tile> P_Tiles; //Cria um vetor de Tiles nas quais pode ter spawn de algum robô

        

        for (const Tile tile : my_tiles) {
            if (tile.can_spawn) {
               P_Tiles.push_back(tile); //Adiciono minhas tiles a P_tiles
            }
        }

        for (const Tile tile : opp_tiles) {
            if (!tile.recycler) {
                targetTiles.push_back(tile);
            }
        }

        for (const Tile tile : neutral_tiles) {
            if (tile.scrap_amount > 0) { //GARANTE QUE UMA TILE NEUTRAL NAO SEJA UMA GRAMA
                targetTiles.push_back(tile);
            }
        }


        for (auto& tile : P_Tiles) { //Para cada uma das minhas tiles
            double distanciaTotal = 0;
            for (const auto& target : targetTiles) {
                distanciaTotal += distancia(tile, target);
            }
            tile.tilePoints = distanciaTotal / 100; //Todas tiles em canSpawnTiles agora terão um score atribuido a elas que é a relacao entre Distancia e a quantidade total de target tiles possíveis
        } 


        sort(P_Tiles.begin(), P_Tiles.end(), [](const Tile& a, const Tile& b) { //Ordena o vetor de canSpawnTiles com base no score de suas tiles;
            return a.tilePoints < b.tilePoints;
        });

        for (Tile tile : my_units) {

            sort(targetTiles.begin(), targetTiles.end(), [&tile](const Tile& a, const Tile& b) { //ORDENA O VETOR DE TARGETS COM BASE NA MENOR DISTANCIA ENTRE UMA TILE DE MYUNITS E UMA TARGET TILE
                return distancia(tile, a) < distancia(tile, b);
            });

        


           /*  bool should_move = false; // TODO: pick whether to move units from here //A FAZER: escolha se eu quero mover minhas tropas a partir da tile atual
            if (should_move) {
                int amount = 0; // TODO: pick amount of units to move //A fazer: escolha a quantidade de unidades a serem movidas
                Tile target; // TODO: pick a destination //A fazer: Escolha um destino
                ostringstream action;
                    action << "MOVE " << amount << " " << tile << " " << target;
                    actions.emplace_back(
                        action.str()
                    );
            } */


            if (!targetTiles.empty()) {
                const Tile& target = targetTiles[0];
                int amount = tile.units > 1 ? tile.units - 1 : 1;
                actions.push_back("MOVE " + to_string(amount) + " " + to_string(tile.x) + " " +
                                  to_string(tile.y) + " " + to_string(target.x) + " " +
                                  to_string(target.y));
            }
        
        
        }
        const Tile& target = my_tiles[0]; 
        if ( my_matter >= 10) { 
            actions.push_back("SPAWN 1 " + to_string(target.x) + " " + to_string(target.y)); //O SPAWN é feito se possuo a quantidade necessaria, e "simplesmente" faz o spawn na primeira tile das minhas tiles a todo momento.
        }

        // Write an action using cout. DON'T FORGET THE "<< endl"
        // To debug: cerr << "Debug messages..." << endl;
        if (actions.empty()) {
             cout << "WAIT" << endl;
        } else {
            for (vector<string>::iterator it = actions.begin(); it != actions.end(); ++it) { //uma vez que todas as acoes foram adicionadas ao vetor:
                cout << *it << ";";
            }
            cout << endl;
        }
    }
}