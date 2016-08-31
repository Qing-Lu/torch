# Windows Build

## Caveats:
- Windows is not officially supported.  These notes are best-effort, supplied by the community, your mileage may vary.  If you find some way(s) of improving these instructions, please go ahead and update this page, and/or post into the unofficial de facto discussion thread at [Add support for compilation on Windows using mingw32](https://github.com/torch/torch7/pull/287#issuecomment-243622035)
- These instructions are a very beta interpretation of the experiences of unsupported community users on this unsupported Windows platform

## Notes from jkjung-avt:

"Specifically my build of Torch7 is with:

- MinGW32 and msys (installed with MinGW Installation Manager)
- CMake 3.6.0 (installed)
- LuaJIT-2.0.4 (built)
- OpenBlas 0.2.18 (built)
- gnuplot-5.0.3 (installed)
- zlib-1.2.8, libpng-1.6.23, jpeg-9b (built) -> these are required by "image"
- Qt 4.8.7 (installed) -> this is required by "qtlua"/"qttorch"
- readline-6.3 (built) -> required by "trepl" (did not work...)

"With the above, I just pulled the latest code from GitHub and built Torch 7. Currently I have the following luarocks packages installed: argcheck, cwrap, dok, env, fftw3, gnuplot, graph, image, lua-cjson, luaffi, luafilesystem, luasocket, nn, nngraph, nnx, optim, paths, penlight, qtlua, qttorch, signal, sundown, sys, torch, trepl, xlua."


