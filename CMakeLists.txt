cmake_minimum_required(VERSION 3.10)
project(OpenCV_C_Project C) # Set project language to C

# Find OpenCV via vcpkg
find_package(OpenCV REQUIRED)

# Add your .c file
add_executable(my_opencv_app image_process.cu)

# Link libraries
target_link_libraries(my_opencv_app PRIVATE ${OpenCV_LIBS})
