#include <iostream>
#include <string>
#include "statpack.h"

int main(int argc, char *argv[]) {
    std::cout << "Running...\n";
    std::string images = "";
    if (argc >= 3) {
        images = argv[1];
    }
    std::cout << "Ending...\n";
    return 0;
}
