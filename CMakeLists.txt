cmake_minimum_required(VERSION 3.21)

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DCUSTOM_DEBUG")

# specify compilers
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set the project name
project(RandomGAN CXX)

# Bundle together
add_subdirectory(mnistLib)
add_subdirectory(app)