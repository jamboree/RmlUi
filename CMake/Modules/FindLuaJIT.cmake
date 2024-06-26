# Try to find the lua library
#  LUAJIT_FOUND - system has lua
#  LUAJIT_INCLUDE_DIR - the lua include directory
#  LUAJIT_LIBRARY - the lua library

FIND_PATH(LUAJIT_INCLUDE_DIR NAMES luajit.h PATH_SUFFIXES luajit luajit-2.0 luajit-2.1)
SET(_LUAJIT_STATIC_LIBS libluajit-5.1.a libluajit.a liblua51.a)
SET(_LUAJIT_SHARED_LIBS luajit-5.1 luajit lua51)
IF(USE_STATIC_LIBS)
	FIND_LIBRARY(LUAJIT_LIBRARY NAMES ${_LUAJIT_STATIC_LIBS} ${_LUAJIT_SHARED_LIBS})
ELSE()
	FIND_LIBRARY(LUAJIT_LIBRARY NAMES ${_LUAJIT_SHARED_LIBS} ${_LUAJIT_STATIC_LIBS})
ENDIF()
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LuaJIT DEFAULT_MSG LUAJIT_LIBRARY LUAJIT_INCLUDE_DIR)
MARK_AS_ADVANCED(LUAJIT_LIBRARY LUAJIT_INCLUDE_DIR)
