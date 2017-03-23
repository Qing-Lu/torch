# Running Torch on Windows

  * [Using a virtual machine](#using-a-virtual-machine)
  * [Binary downloads](#binary-downloads)
  * [Building from sources](#building-from-sources)
    + [Using MinGW](#using-mingw)
    + [Using MSVC automatically](#using-msvc-automatically)
    + [Using Visual Studio manually](#using-visual-studio-manually)
      - [Prerequisites](#prerequisites)
      - [LuaJIT and LuaRocks](#luajit-and-luarocks)
      - [Torch](#torch)
      - [The nn package](#the-nn-package)
      - [The trepl package](#the-trepl-package)
      - [Other packages](#other-packages)
      - [Final notes](#final-notes)
    + [Appendix: Prerequisites](#appendix-prerequisites)
      - [Git](#git)
      - [CMake](#cmake)
      - [MSYS2](#msys2)
      - [MSYS](#msys)
      - [Visual Studio](#visual-studio)
      - [LAPACK](#lapack)
    + [Appendix: Other tools](#appendix-other-tools)
      - [Cmder](#cmder)
      - [ZeroBrane Studio integration](#zerobrane-studio-integration)
    + [Appendix: CI script](#appendix-ci-script)
    + [Appendix: Installing cutorch](#appendix-installing-cutorch)
  * [Known runtime issues with native builds](#known-runtime-issues-with-native-builds)
    + [Loading .t7 files](#loading-t7-files)

<!-- TOC generator: http://ecotrust-canada.github.io/markdown-toc/ Note: screws up non-alnum chars in headers (eg, ':' in appendices) -> verify and fix by hand after re-generating. -->


There are several options to run Torch on Windows. At the moment, the easiest approach is to run Ubuntu inside a virtual machine and install Torch there. This way everything just works, but the downside is that you will quite likely lose GPU acceleration. Another option is to download and try, at your own risk, community-built Windows binaries from the discussion forums. Finally, you can try to build Torch from the sources, but note that this is somewhat tricky at the moment!

> Caveats / work in progress
- Windows is not officially supported.  These notes are best-effort, supplied by the community, your mileage may vary.  If you find some way(s) of improving these instructions, please go ahead and update this page, and/or post into the unofficial de facto discussion thread at [Add support for compilation on Windows using mingw32](https://github.com/torch/torch7/pull/287#issuecomment-243622035)
- These instructions are a very beta interpretation of the experiences of unsupported community users on this unsupported Windows platform

> Todo list
- Try building on Windows 10 using the [Windows Subsystem for Linux](https://msdn.microsoft.com/en-us/commandline/wsl/about)
- GPU support: Add instructions for OpenCL and CUDA versions of everything (blas, torch, nn, ...)
- Build script: We seem to now have an initial process for producing a working build. Writing this knowledge into a build script would make it less painful for new users to build Torch.
  - Automated builds (CI): Once there is a build script, it could be used for setting up a CI server for automatically producing up-to-date Windows binaries.
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


### Using MSVC automatically

Before auto method is fully supported officially, please refer to this customized [distro](https://github.com/BTNC/distro-win) for automatic installation of the whole pack of Torch.


### Using Visual Studio manually

#### Prerequisites

- [Git](#git)
- [CMake](#cmake)
- [Visual Studio](#visual-studio)
- BLAS (eg, [LAPACK](#lapack))
- If building LAPACK from sources: [MSYS2](#msys2)
- Optional: [Cmder](#cmder)

These instructions assume that you use Cmder. You can use the Windows Developer Command Prompt or any other alternative, too.

You can choose between a 32-bit and a 64-bit build by running the following build process in the corresponding environment.
- 32-bit: use the "VS2015 x86" task in Cmder or the VS2015 x86 Native Tools Command Prompt
- 64-bit: use the "VS2015 x64" task in Cmder or the VS2015 x64 Native Tools Command Prompt


#### LuaJIT and LuaRocks

We will install from the main Torch repo while following [diz-vara's build instructions](https://github.com/diz-vara/luajit-rocks) (mostly). You get LuaJIT 2.1 if you add -DWITH_LUAJIT21=ON to both of the cmake commands below, otherwise it will be LuaJIT 2.0. The build commands in the instructions might produce a debug build of LuaJIT, for some odd reason (Torch runs considerably slower with it). -DCMAKE_BUILD_TYPE=Release makes sure that we get a release version.

Choose a directory into which you want to install LuaJIT, LuaRocks and Torch. In the following, it shall be X:\torch\install. (The installation process will install stuff also into ..\share, so an additional directory level helps to keep things clean.) Now, add the directory you chose to your PATH, while making sure that it comes _before_ CMake's bin directory.

Start Cmder and make sure that the correct VS2015 task is active (see above). Cd into some temporary directory, then:

    git clone https://github.com/torch/luajit-rocks.git
    cd luajit-rocks
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=X:/torch/install -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
    nmake
    cmake -DCMAKE_INSTALL_PREFIX=X:/torch/install -G "NMake Makefiles" -P cmake_install.cmake -DCMAKE_BUILD_TYPE=Release

_Note_: The original instructions had an additional -DWIN32=1 option in both cmake commands! Everything seems to work without it, too. (Is it important, anyone?)

Set the following environment variables: (these have been modified a bit from diz-vara's instructions, so as to get some other packages to build)

    set LUA_CPATH = X:/torch/install/?.DLL;X:/torch/install/LIB/?.DLL;?.DLL
    set LUA_DEV = X:/torch/install
    set LUA_PATH = ;;X:/torch/install/?;X:/torch/install/?.lua;X:/torch/install/lua/?;X:/torch/install/lua/?.lua;X:/torch/install/lua/?/init.lua

Test LuaJIT: Completely restart Cmder, then open the Torch task (eg, via the down arrow next to the green plus sign at lower right corner). Try writing some Lua to make sure that the REPL works.

Test LuaRocks: Within Cmder, ctrl-tab back to the VS2015 task and type: luarocks. You should get its help text.


#### Torch

The instructions assume that LuaJIT + LuaRocks was installed to X:\torch\install and LAPACK was installed to X:\torch\lapack. Substitute paths accordingly.

Create the file X:\torch\install\cmake.cmd and add the following content:

    if %1 == -E  (
    cmake.exe  %* 
    ) else (
    cmake.exe -G "NMake Makefiles" -DCMAKE_LINK_FLAGS:implib=libluajit.lib -DLUALIB=libluajit %*
    )

This is from <https://github.com/torch/paths/issues/9> , except -DLUALIB=libluajit has been added (needed to compile the sys package later on).

_Note_: The original file had the additional cmake options -DWIN32=1 and -dLUA_WIN [sic]! Everything seems to work without them, too. (Are they important, anyone?)

Double-check that X:\torch\install is in your PATH _before_ CMake's bin directory.

Now, in Cmder with VS2015 task active, cd into some temporary directory and write:

    luarocks download torch

For some reason, I didn't manage to have CMake auto-detect LAPACK (nor any other BLAS library, for that matter). I proceeded as follows. Open the downloaded rockspec file and add the following to the configuration command (the one starting with `cmake .. -DCMAKE_BUILD_TYPE=Release -DLUA=$(LUA) [...]`), eg, add right before the `&& $(MAKE)` part:

    -DBLAS_LIBRARIES=X:/torch/lapack/lib/libblas.lib -DBLAS_INFO=generic -DLAPACK_LIBRARIES=X:/torch/lapack/lib/liblapack.lib -DLAPACK_FOUND=TRUE

Back in Cmder's VS2015 task:

    git clone git://github.com/torch/torch7.git
    cd torch7
        (git checkout 7bbe17917ea560facdc652520e5ea01692e460d3)
    luarocks make ../torch-scm-1.rockspec

(At the moment, the head version does not compile on Windows. If you get compilation errors, you might want to try the git checkout command above)

Test Torch by launching the Torch task from Cmder and then type:

    require('torch')
    torch.test()

All tests should pass.


#### The nn package

The package depends on luaffi, which in turn fails to build out-of-the-box. Solution is here: <https://github.com/facebook/luaffifb/issues/10>; you need to patch some files.

In Cmder's VS2015 task:

    luarocks download luaffi
    git clone git://github.com/facebook/luaffifb.git
    cd luaffifb

Now open the files ffi.h and test.c, and move the lines

    #include <complex.h>
    #define HAVE_COMPLEX

up into the preceding #else blocks in both files (the #else branch of the #ifdef _WIN32 test).

Back in Cmder's VS2015 task: (_Note_: there is a file with an identical name in the current directory; it won't work, so make sure that you have the `..\` part in the command)

    luarocks make ..\luaffi-scm-1.rockspec

Now you should be able to install the nn package. Again in Cmder's VS2015:

    luarocks install nn

_Note_: At least for me, every compilation unit gives the following kind of warning: `warning C4273: 'THNN_FloatLogSigmoid_updateGradInput': inconsistent dll linkage.`
Any ideas what this is about and how to avoid it?

Now test it by launching Cmder's Torch task and typing:

    require('nn')
    nn.test()

All tests should pass.


#### The trepl package

_Note_: This works only with a 32-bit build.

Install readline for Windows from <http://gnuwin32.sourceforge.net/packages/readline.htm>

Add the GnuWin32 bin directory to your PATH now and completely restart Cmder. (note that having this in your path all the time will mess up some earlier Torch installation steps)

In Cmder's VS2015 task:

    luarocks download trepl
    git clone git://github.com/torch/trepl

Edit the rock file and modify the build/platforms/windows/modules/readline section:

    incdirs = {"windows","C:/Program Files (x86)/GnuWin32/include"},
    libdirs = {"windows","C:/Program Files (x86)/GnuWin32/lib"},
    libraries = {'readline'}

In Cmder's VS2015 task:

    cd trepl
    luarocks make ../trepl-scm-1.rockspec


#### Other packages

_Note_: Tested only with a 32-bit build.

The following packages seem to at least compile/install just fine via luarocks (`luarocks install <packagename>`):

> luafilesystem, inspect, image, nnx, optim, gnuplot

The sys package: This installs without issues, as long as you have -DLUALIB=libluajit added to your cmake.cmd (see earlier).

The dataset and ipc packages fail non-trivially, due to pthreads dependency.


#### Final notes

The preceding process always installs the most recent version of everything. Just in case that something has changed since writing this and broke something, here are the versions that were used and are known to work:

> https://github.com/torch/luajit-rocks/commit/4eb4c5b6c6cf94badadebc8d5c39a1d470950036
https://github.com/torch/torch7/commit/7c740d5e8ec7fc10edbc3a75f2667e481eb47180
https://github.com/torch/paths/commit/68d579a2d3b1b0bb03a11637632e6e699b14ad80
https://github.com/torch/cwrap/commit/dbd0a623dc4dfb4b8169d5aecc6dd9aec2f22792
https://github.com/facebook/luaffifb/commit/d1c9712bfaaa73f9bc064227f120320e97fff517
https://github.com/torch/nn/commit/07d3bdd496be72dd132eb70eab96478b96547ffe


Source: https://groups.google.com/d/topic/torch7/iQpNfJB_oy0/discussion

See also: https://groups.google.com/d/topic/torch7/h4wDj23Ap7c/discussion



### Appendix: Prerequisites

This section contains detailed install instructions for some of the prerequisites. You might not need all of these; please refer to the instructions above.


#### Git

PortableGit or Git for Windows are some possible options. Cmder seems to come with a git client, too. Install one, then make sure that you have the git executable in your PATH.


#### CMake

Install CMake as usual, then make sure that you have it's executable in your PATH.


#### MSYS2

The 64-bit version of MSYS2 provides toolchains for both 32-bit and 64-bit targets, so we are going to use it. Use the 64-bit installer (msys2-x86_64) from the [MSYS2 site](https://msys2.github.io/). As usual with tools from the other side, don't install to Program Files (due to spaces).

Follow the instructions at the MSYS2 site for updating everything.
- If prompted to force close the window and restart MSYS2, you might need to also manually kill pacman.exe via task manager.
- The update may break your MSYS2 Shell shortcuts. Fix it by editing the shortcut files: for MSYS2 Shell, change the suffix of the target from .bat to .cmd; for MinGW-w32 MinGW-w64 Shells, use the same cmd as with MSYS2 but add the options -mingw32 or -mingw64 after it.

Now, install some packages. Open the MSYS2 Shell and enter the following commands. The x86-64 packages are for 64-bit targets and the i686 packages are for 32-bit targets. Choose what you need or install both.

    pacman -S git tar make
        (don't install cmake, it won't work; you need to use the native Windows version of cmake)
    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran
    pacman -S mingw-w64-i686-gcc mingw-w64-i686-gcc-fortran

Notes:
- The MSYS2 MSYS environment can be started via the MSYS2 Shell shortcut. Use this for package management etc. Don't use this for compiling! 
- The 64-bit toolchain can be used via the MinGW-w64 Win64 Shell shortcut.
- The 32-bit toolchain can be used via the MinGW-w64 Win32 Shell shortcut.
- You might _not_ want to add the bin directory to your PATH


#### MSYS

_Note_: [MSYS2](#msys2) provides both 32-bit and 64-bit toolchains, so you might want to use that in both cases!

Use the MinGW installer from <http://www.mingw.org/> and install all meta-packages from the "Basic Setup" section. As usual with tools from the other side, don't install to Program Files (due to spaces).

Add `C:\MinGW /mingw` (or whatever path you chose) to the MSYS etc/fstab file, as instructed at <http://www.mingw.org/wiki/getting_started> -> After Installing You Should ...

Notes:
- The package manager is at bin/mingw-get.exe
- The MSYS environment can be started via msys\1.0\msys.bat
- You might _not_ want to add the bin directory to your PATH


#### Visual Studio

Visual Studio Community 2015 seems to work just fine. During installation, choose Custom Installation and check the C/C++ tools. When entering build commands, use one of the following shortcuts (installed by VS):
- VS2015 x86 Native Tools Command Prompt (32-bit toolchain)
- VS2015 x64 Native Tools Command Prompt (64-bit toolchain)

or use an alternative like [Cmder](#cmder).


#### LAPACK

Following the instructions at <http://icl.cs.utk.edu/lapack-for-windows/lapack/#build> (loosely). Note that using the GNUtoMS option requires that you have Visual Studio installed.

Download and unzip <http://www.netlib.org/lapack/lapack-3.6.1.tgz> (or newer). Note that the site provides a prebuilt version too, but it might be slower on your machine (also, for me, VS recognized the prebuilt 64-bit lib as 32-bit).

If you are building 64-bit libraries, then open the [MSYS2](#msys2) MinGW-w64 Win64 Shell. For 32-bit libraries, open the [MSYS2](#msys2) MinGW-w64 Win32 Shell. Make sure that the command prompt indicates the correct toolchain, either `MINGW64` or `MINGW32`. Cd into the extracted package, then:

    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=X:/torch/lapack -G "MSYS Makefiles" -DBUILD_SHARED_LIBS=1 -DCMAKE_GNUtoMS=1
        (you might have to use the full cmake path)
    make
    make install

To proceed with Torch installation, make sure that you have the following DLLs in your path. You can copy then all to the directory into which you are going to install Torch, then add that directory to your PATH.
- All DLLs from X:\torch\lapack\bin\
- The following DLLs from the bin directory of your MSYS2 installation:
  - For 32-bit libraries: libgcc_s_dw2-1.dll
  - For 64-bit libraries: libgcc_s_seh-1.dll
  - For both 64-bit and 32-bit libraries: libgfortran-3.dll, libquadmath-0.dll, libwinpthread-1.dll (Note: if you installed both toolchains, then there will exist 64-bit and 32-bit versions of these; make sure to copy the correct ones!)



### Appendix: Other tools


#### Cmder

You might want to use Cmder to replace the Windows Developer Command Prompt to make things less painful. The Torch REPL can be run through it, too.

1. Download and install Cmder, or Cmder Lite if you already have another Git client installed.

1. 32-bit toolchain: Create a new task for VS: Go to Settings -> Startup -> Tasks and create a new task. Name it "VS2015 x86" or something and add the following string as the startup command (replace the project path with whatever you have):
  `cmd /k ""%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" & "%ConEmuDir%\..\init.bat"" -new_console:d:"X:\work":t:"VS2015 x86"`

1. 64-bit toolchain: Repeat as above, except name the task "VS2015 x64" or something and use the following startup command:
  `cmd /k ""%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" amd64 & "%ConEmuDir%\..\init.bat"" -new_console:d:"X:\work":t:"VS2015 x64"`

1. Create a new task for Torch: Go to Settings -> Startup -> Tasks and create a new task. Name it Torch or something and add the following string as the startup command (replace the paths with whatever you are going to use):
  `X:\torch\install\luajit.exe -new_console:d:"X:\work\torch_projects":t:"Torch"`

You might want to make the toolchain that you intend to use the default one, so as to avoid accidentally compiling with the wrong one.

Aliases can be added as usual (they persist restarts): `alias ll=ls -la --show-control-chars -F --color $*`
Adding Sublime Text 3's program directory to your path lets you use the command "subl" to quickly open files (for deeper integration, see eg http://goo.gl/yF173L ).

Note: There is the option "Inject ConEmuHk" under Settings -> Features. Enabling it slows down all command execution, especially compilation. Disabling it messes up colored output for second level processes. You might want to switch this temporarily on while trying to make sense of colored make output and otherwise keep it off.

Then some patching: command line wrapping might be broken when inside git repos. To solve this, open the file `C:\Program Files (x86)\cmder_mini\config\cmder.lua` and comment out the line with the os.execute() call. (see https://github.com/cmderdev/cmder/issues/749 )


#### ZeroBrane Studio integration

The instructions at <http://notebook.kulchenko.com/zerobrane/torch-debugging-with-zerobrane-studio> seem to work, except that you might have to point ZBS to the LuaJIT executable instead of the installation directory: `path.torch = [[X:/torch/install/luajit.exe]]`

The examples worked fine when executed from a script file, but using Torch from the ZBS's REPL did not work well (ZBS REPL seems to ignore the Lua interpreter setting and consequently uses the built-in, older version of LuaJIT).

Starting a debugging session might complain about a missing lua51.dll. This is simply a naming issue: ZBS looks for lua51.dll, while we have libluajit.dll. A quick fix is to copy, in the Torch installation directory, libluajit.dll to lua51.dll.
WARNING: This will work here, apparently because the dlls are used from different processes, but in general this is probably a bad solution. (Managing to load both dlls in a single application would lead to two different address spaces, if I understand correctly, and then to trouble. Does anyone have ideas for solving this in a better way?)

Remote debugging (<https://studio.zerobrane.com/doc-remote-debugging>) seems to work, too, as long as you make sure that mobdebug.lua from C:\ZeroBraneStudio\lualibs\mobdebug is in your Lua search path (eg, copy it to the `lua` subdirectory in your Torch install directory).

Debugging a 64-bit Torch build using ZBS seems to work just fine, too. You just need a 64-bit version of the socket library, which can be found [here](https://github.com/pkulchenko/ZeroBraneStudio/issues/500#issuecomment-122347792). Extract it under X:\torch\install\lib. You might need to copy also socket.lua from the ZBS installation, in addition to mobdebug.lua, into X:\torch\install\lua.

### Appendix: CI Script

There is a CI script at: https://github.com/hughperkins/distro-cl/blob/distro-win/build_windows.bat

(And a script to install msys64 at https://github.com/hughperkins/distro-cl/blob/distro-win/installdeps_win.bat )

This will build the base `torch7` module, and run its unit tests.  In the future it could plausibly be extended to handle `nn`, and so on.


### Appendix: Installing cutorch

I'm keeping this separate from the above sections as this was my experience in installing, in particular, cutorch, after getting Lua, Torch and some other packages (like nn & nnx) installed.

Note that the following instructions were performed for a 64-bit Luarocks version  installed in the directory `C:\luajit-rocks` on my machine.

 - Direct a command prompt to the `C:\luajit-rocks\luainstall` directory

 - Typing `luarocks install cutorch` failed when implementing a cmake command, b/c the command is using a Linux -j argument related to the number of processors.

 - To get around this, do the following (I got this idea online from 'smth chntla'):

    - cd into the C:\luajit-rocks\luainstall\luarocks folder

    - clone the cutorch source files with the command `git clone https://github.com/torch/cutorch`

    - Then edit the file `C:\luajit-rocks\luainstall\luarocks\cutorch\rocks\cutorch-scm-1.rockspec` by editing the cmake lines and deleting the -j options that specifies the number of processors (it is not critically important)

 - Next step is, back in the command prompt, cd to the C:\luajit-rocks\luainstall\luarocks\cutorch directory and type `luarocks make rocks/cutorch-scm-1.rockspec`

 - However, this gave an error related to the a.lib and a.exp files.  

 - To overcome, do the following (got this online from Siavesh Gorji):

    - Open up the CMake Gui

    - In the source code entry put: `C:\luajit-rocks\luainstall\luarocks\cutorch`

    - Build the binaries at: `C:\luajit-rocks\luainstall\luarocks\cutorch\build`

    - Hit 'Configure' (you may get some warnings but should not get any errors)

    - Hit 'Generate' to generate the makefile you need.

    - Then go back to the command prompt and cd to C:\luajit-rocks\luainstall\luarocks\cutorch and run `luarocks make ./rocks/cutorch-scm-1.rockspec`   <= it takes about 10 minutes but cutorch should build.

I hope the above helps someone navigate these waters - as it took me some time (and a lot of help from folks on the discussion boards!) to figure it out.

## Known runtime issues with native builds

### Loading .t7 files

You may encounter issues loading binary .t7 data files, because of differences in interpreting various data types as 32-bit or 64-bit.  You might be able to solve these by replacing `torch.load('somefile.t7')` with `torch.load('somefile.t7', 'b64')`
