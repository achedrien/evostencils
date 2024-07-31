# Installation de Guix

Ouvrir un terminal et taper les commandes suivantes :
`cd /tmp` (ou autre)
`wget https://git.savannah.gnu.org/cgit/guix.git/plain/etc/guix-install.sh` 
`chmod +x guix-install.sh`
`./guix-install.sh`

# Installation de composyx

Ouvrir le fichier channel configuration (`~/.config/guix/channels.scm`) sinon le créer. Ecrire les commandes suivantes dans le fichier :
`(cons (channel
        (name 'guix-hpc-non-free)
        (url "https://gitlab.inria.fr/guix-hpc/guix-hpc-non-free.git"))
      %default-channels)`
Dans le terminal taper la commande `guix pull` pour mettre à jour guix (cela prend un moment).

# Compiler un code utilisant composyx

## Création d'un fichier `CMakeLists.txt`

Créer un fichier `CMakeLists.txt` contenant :

`cmake_minimum_required(VERSION 3.12)

project(COMPOSYX_EXAMPLE CXX C Fortran)

find_package(maphyspp REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(exec main.cpp fonction1.cpp fonction2.cpp)

target_link_libraries(exec PRIVATE MAPHYSPP::maphyspp)
`
## Compilation 

Pour ouvrir un terminal un terminal avec composyx installé il faut taper la commande :
`guix shell --pure maphys++ -D maphys++ coreutils ncurses bash -- bash --norc`
Se placer à l'endroit où se trouve les fichiers `.cpp` et `CMakeLists.txt`
Pour compiler :
`cmake -B build
cmake --build build --target exec`

## Execution

build/exec






