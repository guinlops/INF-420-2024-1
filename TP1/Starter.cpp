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
    int x, y, scrap_amount, owner, units;
    bool recycler, can_build, can_spawn, in_range_of_recycler;
        ostream& dump(ostream& ioOut) const {
        ioOut << x << " " << y;
        return ioOut;
    }
};
ostream& operator<<(ostream& ioOut, const Tile& obj) { return obj.dump(ioOut); }


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

        
        for (vector<Tile>::iterator it = my_tiles.begin(); it != my_tiles.end(); ++it) { //Cria um iterador para percorrer o vector de minhas tiles;
        //Já que cada tile contem um objeto, o iterador é um objeto também.
            Tile tile = *it;
            if (tile.can_spawn) {
                int amount = 0; // TO DO: pick amount of robots to spawn here // A fazer: escolher a quantidade de robos para serem spawnados na Minha tile:
                if (amount > 0) {
                    ostringstream action; //
                    action << "SPAWN " << amount << " " << tile; //string SPAWN.... é guardada em action
                    actions.emplace_back(
                        action.str() // action como cout
                    );
                }
            }
            if (tile.can_build) {
                bool should_build = false; // TODO: pick whether to build recycler here// A fazer: se eu quiser construir um recycler no tile atual:
                if (should_build) {
                    ostringstream action;
                    action << "BUILD " << tile;
                    actions.emplace_back(
                        action.str()
                    );
                }
            }
        }

        for (Tile tile : my_units) {
            bool should_move = false; // TODO: pick whether to move units from here //A FAZER: escolha se eu quero mover minhas tropas a partir da tile atual
            if (should_move) {
                int amount = 0; // TODO: pick amount of units to move //A fazer: escolha a quantidade de unidades a serem movidas
                Tile target; // TODO: pick a destination //A fazer: Escolha um destino
                ostringstream action;
                    action << "MOVE " << amount << " " << tile << " " << target;
                    actions.emplace_back(
                        action.str()
                    );
            }
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