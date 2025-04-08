cmake_minimum_required (VERSION 3.20)
set (CMAKE_CXX_STANDARD 20)

project (my_nlp)
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE "src/run.cpp")
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}")

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS json math) 

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

add_subdirectory("src/data_processing")
add_subdirectory("src/math")
add_subdirectory("src/models")
add_subdirectory("src/pipline_runner")
install(TARGETS ${PROJECT_NAME})