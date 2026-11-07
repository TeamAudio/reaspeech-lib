cmake_minimum_required(VERSION 3.21)
project(reaspeech_sdk_smoke LANGUAGES C CXX)
find_package(ReaSpeech CONFIG REQUIRED HINTS "${CMAKE_CURRENT_LIST_DIR}/../..")
foreach(language IN ITEMS c cpp)
  add_executable(smoke_${language} smoke.${language})
  target_link_libraries(smoke_${language} PRIVATE ReaSpeech::reaspeech)
  if(WIN32)
    add_custom_command(TARGET smoke_${language} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${ReaSpeech_DIR}/../../../bin" "$<TARGET_FILE_DIR:smoke_${language}>")
  endif()
  add_test(NAME smoke_${language} COMMAND smoke_${language})
endforeach()
enable_testing()
