# Running Torch on Windows

  * [Using a virtual machine](#using-a-virtual-machine)
  * [Binary downloads](#binary-downloads)
  * [Building from sources](#building-from-sources)
    + [Using MinGW](#using-mingw)
    + [Using Visual Studio](#using-visual-studio)
  * [Known run-time issues](#known-run-time-issues)
    + [Loading .t7 files](#loading-t7-files)

<!-- TOC generator: http://ecotrust-canada.github.io/markdown-toc/ -->


There are several options to run Torch on Windows. At the moment, the easiest approach is to run Ubuntu inside a virtual machine and install Torch there. This way everything just works, but the downside is that you will quite likely lose GPU acceleration. Another option is to download and try, at your own risk, community-built Windows binaries from the discussion forums. Finally, you can try to build Torch from the sources, but note that this is somewhat tricky at the moment!

> Caveats / work in progress
- Windows is not officially supported.  These notes are best-effort, supplied by the community, your mileage may vary.  If you find some way(s) of improving these instructions, please go ahead and update this page, and/or post into the unofficial de facto discussion thread at [Add support for compilation on Windows using mingw32](https://github.com/torch/torch7/pull/287#issuecomment-243622035)
- These instructions are a very beta interpretation of the experiences of unsupported community users on this unsupported Windows platform

> Todo list
- Expand the build instructions
- Build script: We seem to now have an initial process for producing a working build. Writing this knowledge into a build script would make it less painful for new users to build Torch.
  - Automated builds (CI): Once there is a build script, it could be used for setting up a CI server for automatically producing up-to-date Windows binaries.
- 64-bit build: The current process is for producing a 32-bit build. Someone could try it out with a toolchain supporting 64-bit targets (MSYS2 / MinGW-w64, Visual Studio, .. ?)
- Dealing with CPU extensions: The Torch build process auto-detects, by default, the CPU extensions (SSE, AVX, ..) supported by the host CPU and enables corresponding optimizations. Distributing the resulting binary might lead to crashes for users with older CPUs. (some options in the thread)
- (VM) GPU acceleration with a VM install: PCIe passthrough...is this currently possible with any VM software on a Windows host?
- (VM) Machine images: Are there any pre-made Torch VM images? How about the Docker images at [Cheatsheet](https://github.com/torch/torch7/wiki/Cheatsheet#installing-and-running-torch)?

>If you have the time and skills for some of these tasks, then please, do contribute!

>Discussion thread: [Add support for compilation on Windows using mingw32](https://github.com/torch/torch7/pull/287#issuecomment-243622035)



## Using a virtual machine

You can use VirtualBox or some other virtualization software to set up Ubuntu and then simply install Torch onto that by following the [official Ubuntu install instructions](http://torch.ch/docs/getting-started.html#_). However, GPU acceleration requires PCIe passthrough, which can be something from difficult to impossible to get working on Windows.



## Binary downloads

There are some community-built Windows binaries available at least here:
- [Add support for compilation on Windows using mingw32](https://github.com/torch/torch7/pull/287#issuecomment-240965177)
- [Using cutorch and cunn on windows](https://groups.google.com/d/msg/torch7/A5XUU4u9Tjw/85D19tzkR7AJ)

**Note**: These are _not_ official builds and they are not verified in any way by the Torch maintainers. Use at your own risk!



## Building from sources


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
