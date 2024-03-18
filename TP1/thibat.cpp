#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

const int ME = 1;
const int OPP = 0;
const int NONE = -1;

struct Tile {
    int x, y, scrapAmount, owner, units;
    bool recycler, canBuild, canSpawn, inRangeOfRecycler;
    double spawnScore;
};

int distance(const Tile& a, const Tile& b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

int main() {
    int width, height;
    std::cin >> width >> height;
    std::cin.ignore();

    while (true) {
        std::vector<Tile> tiles, myUnits, oppUnits, myRecyclers, oppRecyclers, oppTiles, myTiles, neutralTiles;
        int myMatter, oppMatter;
        std::cin >> myMatter >> oppMatter;
        std::cin.ignore();

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int scrapAmount, owner, units;
                std::string recyclerStr, canBuildStr, canSpawnStr, inRangeOfRecyclerStr;
                std::cin >> scrapAmount >> owner >> units >> recyclerStr >> canBuildStr >> canSpawnStr >> inRangeOfRecyclerStr;
                std::cin.ignore();

                bool recycler = recyclerStr == "1";
                bool canBuild = canBuildStr == "1";
                bool canSpawn = canSpawnStr == "1";
                bool inRangeOfRecycler = inRangeOfRecyclerStr == "1";

                Tile tile = {x, y, scrapAmount, owner, units, recycler, canBuild, canSpawn, inRangeOfRecycler};
                tiles.push_back(tile);

                if (tile.owner == ME) {
                    myTiles.push_back(tile);
                    if (tile.units > 0) {
                        myUnits.push_back(tile);
                    } else if (tile.recycler) {
                        myRecyclers.push_back(tile);
                    }
                } else if (tile.owner == OPP) {
                    oppTiles.push_back(tile);
                    if (tile.units > 0) {
                        oppUnits.push_back(tile);
                    } else if (tile.recycler) {
                        oppRecyclers.push_back(tile);
                    }
                } else {
                    neutralTiles.push_back(tile);
                }
            }
        }

        std::vector<std::string> actions;
        std::vector<Tile> targetTiles;

        for (const auto& tile : oppTiles) {
            if (!tile.recycler) {
                targetTiles.push_back(tile);
            }
        }

        for (const auto& tile : neutralTiles) {
            if (tile.scrapAmount > 0) {
                targetTiles.push_back(tile);
            }
        }

        std::vector<Tile> canSpawnTiles;

        for (const auto& tile : myTiles) {
            if (tile.canSpawn) {
                canSpawnTiles.push_back(tile);
            }
        }

        for (auto& tile : canSpawnTiles) {
            double totalDistance = 0;
            for (const auto& target : targetTiles) {
                totalDistance += distance(tile, target);
            }
            tile.spawnScore = totalDistance / targetTiles.size();
        }

        std::sort(canSpawnTiles.begin(), canSpawnTiles.end(), [](const Tile& a, const Tile& b) {
            return a.spawnScore < b.spawnScore;
        });

        const Tile& target = canSpawnTiles[0];
        if (!canSpawnTiles.empty() && myMatter >= 10) {
            actions.push_back("SPAWN 1 " + std::to_string(target.x) + " " + std::to_string(target.y));
        }

        for (const auto& tile : myUnits) {
            std::sort(targetTiles.begin(), targetTiles.end(), [&tile](const Tile& a, const Tile& b) { //ORDENA O VETOR DE TARGETS COM BASE NA MENOR DISTANCIA ENTRE UMA TILE DE MYUNITS E UMA TARGET TILE
                return distance(tile, a) < distance(tile, b);
            });

            if (!targetTiles.empty()) {
                const Tile& target = targetTiles[0];
                int amount = tile.units > 1 ? tile.units - 1 : 1;
                actions.push_back("MOVE " + std::to_string(amount) + " " + std::to_string(tile.x) + " " +
                                  std::to_string(tile.y) + " " + std::to_string(target.x) + " " +
                                  std::to_string(target.y));
            }
        }

        if (!actions.empty()) {
            for (const auto& action : actions) {
                std::cout << action << ";";
            }
        } else {
            std::cout << "WAIT";
        }

        std::cout << std::endl;
    }

    return 0;
}