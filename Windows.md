# Running Torch on Windows

  * [Using a virtual machine](#using-a-virtual-machine)
  * [Binary downloads](#binary-downloads)
  * [Building from sources](#building-from-sources)
    + [Caveats](#caveats)
    + [Using MinGW](#using-mingw)
    + [Using Visual Studio](#using-visual-studio)
  * [Known run-time issues](#known-run-time-issues)
    + [Loading .t7 files](#loading-t7-files)

<!-- TOC generator: http://ecotrust-canada.github.io/markdown-toc/ -->


There are several options to run Torch on Windows. At the moment, the easiest approach is to run Ubuntu inside a virtual machine and install Torch there. This way everything just works, but the downside is that you will quite likely lose GPU acceleration. Another option is to download and try, at your own risk, community-built Windows binaries from the discussion forums. Finally, you can try to build Torch from the sources, but note that this is somewhat tricky at the moment!



## Using a virtual machine

You can use VirtualBox or some other virtualization software to set up Ubuntu and then simply install Torch onto that. However, GPU acceleration requires PCIe passthrough, which can be something from difficult to impossible to get working on Windows.

(PCIe passthrough on a Windows host: is it currently possible with any VM software?)

(Docker images? VirtualBox images?)



## Binary downloads

There are some community-built Windows binaries available at least here:
- [Add support for compilation on Windows using mingw32](https://github.com/torch/torch7/pull/287#issuecomment-240965177)
- [Using cutorch and cunn on windows](https://groups.google.com/d/msg/torch7/A5XUU4u9Tjw/85D19tzkR7AJ)

**Note**: These are _not_ official builds and they are not verified in any way by the Torch maintainers. Use at your own risk!

(32 vs. 64 bit builds?)

(SSE, AVX, ..: Binaries built for a CPU that supports certain extensions will crash on older CPUs. Ideal solution: use a compiler that supports CPU dispatching. At least the Intel compiler should be able to do this.)



## Building from sources


### Caveats
- Windows is not officially supported.  These notes are best-effort, supplied by the community, your mileage may vary.  If you find some way(s) of improving these instructions, please go ahead and update this page, and/or post into the unofficial de facto discussion thread at [Add support for compilation on Windows using mingw32](https://github.com/torch/torch7/pull/287#issuecomment-243622035)
- These instructions are a very beta interpretation of the experiences of unsupported community users on this unsupported Windows platform


### Using MinGW

Notes from jkjung-avt: "Specifically my build of Torch7 is with:

- MinGW32 and msys (installed with MinGW Installation Manager)
- CMake 3.6.0 (installed)
- LuaJIT-2.0.4 (built)
- OpenBlas 0.2.18 (built)
- gnuplot-5.0.3 (installed)
- zlib-1.2.8, libpng-1.6.23, jpeg-9b (built) -> these are required by "image"
- Qt 4.8.7 (installed) -> this is required by "qtlua"/"qttorch"
- readline-6.3 (built) -> required by "trepl" (did not work...)

With the above, I just pulled the latest code from GitHub and built Torch 7. Currently I have the following luarocks packages installed: argcheck, cwrap, dok, env, fftw3, gnuplot, graph, image, lua-cjson, luaffi, luafilesystem, luasocket, nn, nngraph, nnx, optim, paths, penlight, qtlua, qttorch, signal, sundown, sys, torch, trepl, xlua."


### Using Visual Studio

( https://groups.google.com/d/topic/torch7/iQpNfJB_oy0/discussion )

( https://groups.google.com/d/topic/torch7/h4wDj23Ap7c/discussion )



## Known run-time issues

### Loading .t7 files

You may encounter issues loading binary .t7 data files, because of differences in interpreting various data types as 32-bit or 64-bit.  You might be able to solve these by replacing `torch.load('somefile.t7')` with `torch.load('somefile.t7', 'b64')`
