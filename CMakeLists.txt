cmake_minimum_required(VERSION 3.0.2)
project(mediapipe_ros)


find_package(
  catkin REQUIRED
  catkin_virtualenv
  dynamic_reconfigure
  message_generation
)

#catkin_python_setup()

add_message_files(
  FILES
  Hand.msg
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_generate_virtualenv(
  INPUT_REQUIREMENTS requirements.in
  PYTHON_INTERPRETER python3.7
  USE_SYSTEM_PACKAGES FALSE
  ISOLATE_REQUIREMENTS TRUE
  CHECK_VENV FALSE
)

file(GLOB SCRIPTS_FILES "scripts/*.py")
catkin_install_python(
  PROGRAMS ${SCRIPTS_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )


install(FILES requirements.in requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
