A quick page for everything Torch.

* [Newbies](#newbies)
* [Installing and Running Torch](#installing-and-running-torch)
* [Installing Packages](#installing-packages)
* [Tutorials, Demos by Category](#tutorials-demos-by-category)
* [Loading popular datasets](#loading-popular-datasets)
* [List of Packages by Category](#list-of-packages-by-category)
  * [General Math](#general-math)
  * [Data formats](#data-formats)
  * [Machine Learning](#machine-learning)
  * [Visualization](#visualization)
  * [Computer Vision](#computer-vision)
  * [Images](#images)
  * [Videos](#videos)
  * [Audio](#audio)
  * [Natural Language Processing](#natural-language-processing)
  * [Sensor Input/Output](#sensor-inputoutput)
  * [Distributed Computing / Parallel Processing](#distributed-computing--parallel-processing)
  * [Alternative REPLs](#alternative-repls)
  * [Interfaces to third-party libraries](#interfaces-to-third-party-libraries)
  * [Asynchronous paradigm - (like nodejs)](#asynchronous-paradigm---like-nodejs)
  * [Networking](#networking)
  * [Security](#security)
  * [CUDA](#cuda)
  * [OpenCL](#opencl)
  * [Miscellaneous](#miscellaneous)
* [Creating your own package](#creating-your-own-package)
* [Debuggers / Visual Debuggers / IDEs](#debuggers--visual-debuggers--ides)
* [GPU Support](#gpu-support)
* [Gotchas](#gotchas)

Newbies
=======
1. Read this page end-to-end (especially the Gotchas)
2. Install torch
3. Look at the intro tutorial
4. Play around with the interpreter
5. Create your own script
6. Create your own package
7. Contribute! :)

Installing and Running Torch
============================
OSX and Ubuntu
---------------
Run the commands to install Torch globally:
```
curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-luajit+torch | bash
luarocks install env
```

Run torch using the command
```
luajit -lenv
```
If you need QT visualization, run torch using the command
```
qlua
```
If you need gfx.js visualization, run torch using the following command
```
luajit -lgfx.go
```

iOS
---------------
* https://github.com/clementfarabet/torch-ios

Android
---------------
* https://github.com/soumith/torch-android

Installing Packages
===================
Given a package name, you can install it at your terminal with:
```
luarocks install [packagename]
```

Loading popular datasets
============================
* [MNIST Loader](https://github.com/andresy/mnist) - by Ronan Collobert
* [torch-datasets](https://github.com/rosejn/torch-datasets) - Scripts to load several popular datasets including:
  * BSR 500
  * CIFAR-10
  * COIL
  * Street View House Numbers
  * MNIST
  * NORB

Also look at the forks of this repository, if anyone added support for more datasets in their branches.

Tutorials, Demos by Category
===================================
Tutorials
---------
* http://code.cogbits.com/wiki/doku.php?id=start

Demos
------
* [Original demos repository, lots of demos](https://github.com/clementfarabet/torch7-demos)
* [Better maintained and more updated repository by Purdue e-Lab](https://github.com/e-lab/torch7-demos)
* [Another face detector by @jonathantompson](https://github.com/jonathantompson/geiger_facedetector)
* [Some body tracking stuff by @jonathantompson](https://github.com/jonathantompson/cbody)
* [Galaxy-Zoo training (CUDA demo)](https://github.com/soumith/galaxyzoo)
* [Recpool](https://github.com/soumith/recpool) - Reconstructing pooling networks by Jason Rolfe, this branch has some fixes.
* [Music Tagging](https://github.com/mbhenaff/MusicTagging) - Music Tagging scripts for torch7 by Mikael Henaff

List of Packages by Category
============================
General Math
------------
* [torch](https://github.com/torch/torch7) - The core torch package. Apart from tensor operations, has convolutions, cross-correlations, basic linear algebra operations, eigen values/vectors etc.
* [cephes](http://jucor.github.io/torch-cephes) - Cephes mathematical functions library, wrapped for Torch. Provides and wraps the 180+ special mathematical functions from the Cephes mathematical library, developed by Stephen L. Moshier. It is used, among many other places, at the heart of SciPy.
* [graph](https://github.com/torch/graph) - Graph package for Torch
* [randomkit](http://jucor.github.io/torch-randomkit/) - Numpy's randomkit, wrapped for 
* [signal](http://soumith.ch/torch-signal/signal/) - A signal processing toolbox for Torch-7. FFT, DCT, Hilbert, cepstrums, stft

Data formats
------------
* [csvigo](https://github.com/clementfarabet/lua---csv) - A CSV library, for Torch
* [hdf5](https://github.com/d11/torch-hdf5) - Read and write Torch tensor data to and from Hierarchical Data Format files.
* [lua-cjson](http://www.kyne.com.au/~mark/software/lua-cjson.php) - A fast JSON encoding/parsing module
* [mattorch](https://github.com/clementfarabet/lua---mattorch) - An interface between Matlab and Torch
* LuaXML - a module that maps between Lua and XML without much ado
* LuaZip - Library for reading files inside zip files
* MIDI - Reading, writing and manipulating MIDI data
* [audio](https://github.com/soumith/lua---audio/) - Loads as Tensors, all audio formats supported by libsox (mp3, wav, aac, ogg, flac, etc.)
* [csv2torchdatasets](https://github.com/andreirusu/csv2torch-datasets) - Simple Torch7 tool for converting Kaggle-style csv files to torch-datasets.

Machine Learning
------------
* manifold - A package to manipulate manifolds
* nn - Neural Network package for Torch
* nngraph - This package provides graphical computation for nn library in Torch7.
* nnx - A completely unstable and experimental package that extends Torch's builtin nn library
* optim - An optimization library for Torch.
* svm - Torch-SVM library
* unsup - A package for unsupervised learning in Torch. Provides modules that are compatible with nn (LinearPsd, ConvPsd, AutoEncoder, ...), and self-contained algorithms (k-means, PCA). 
* [lbfgs](https://github.com/clementfarabet/lbfgs) - FFI Wrapper for liblbfgs
* [vowpalwabbit](https://github.com/clementfarabet/vowpal_wabbit) - An old vowpalwabbit interface to torch.
* [OpenGM](https://github.com/clementfarabet/lua---opengm) - OpenGM is a C++ library for graphical modeling, and inference. The Lua bindings provide a simple way of describing graphs, from Lua, and then optimizing them with OpenGM.
* [sphagetti](https://github.com/MichaelMathieu/lua---spaghetti) - Spaghetti (sparse linear) module for torch7 by @MichaelMathieu
* [LuaSHKit](https://github.com/ocallaco/LuaSHkit) - A lua wrapper around the Locality sensitive hashing library SHKit
* [kernel smoothing](https://github.com/rlowrance/kernel-smoothers) - KNN, kernel-weighted average, local linear regression smoothers

Visualization
------------
Mainly provided by two styles:
* gfx.js - A graphics backend for the browser, with a Torch7 client. Extend this by writing simple html/javascript templates

or

* qtlua - Powerful QT interface to Lua
* qttorch - QT interface to Torch

* gnuplot - Torch interface to Gnuplot


Computer Vision
------------
* fex - A package for feature extraction in Torch. Provides SIFT and dSIFT modules. 
* imgraph - An image/graph library for Torch. This package provides routines to construct graphs on images, segment them, build trees out of them, and convert them back to images.
* videograph - A video/graph library for Torch. This package provides routines to construct graphs on videos, segment them, build trees out of them, and convert them back to videos.
* [opencv wrapper](https://github.com/marcoscoffier/lua---opencv) by @marcoscoffier
* [saliency](https://github.com/marcoscoffier/torch-saliency) - code and tools around integral images. A library for finding interest points based on fast integral histograms. 
* [stitch](https://github.com/marcoscoffier/lua---stitch) - allows us to use hugin to stitch images and apply same stitching to a video sequence
* [sfm](https://github.com/marcoscoffier/lua---sfm) - A bundle adjustment/structure from motion package
* [optical-flow](https://github.com/marcoscoffier/optical-flow) - This is a simple wrapper around the optical-flow algorithm developed/published by C.Liu
* [depth-estimation](https://github.com/MichaelMathieu/depth-estimation) - Depth estimation scripts by @MichaelMathieu
* [depth-estimation2](https://github.com/MichaelMathieu/depth-estimation2) - Depth estimation scripts by @MichaelMathieu
* [OpenCV 2.4](https://github.com/MichaelMathieu/lua---opencv24) - a simple wrapper for certain funcs from the OpenCV library, version 2.4
* [sfm2](https://github.com/MichaelMathieu/lua---sfm2) - OpenCV based SFM functions for Torch
* [OverFeat](https://github.com/sermanet/OverFeat) - A quick feature extractor based on Overfeat with pretty clunky torch bindings
* [Overfeat-torch](https://github.com/jhjin/overfeat-torch) - Better torch bindings for Torch by @jhjin


Images
------------
* image - An image library for Torch. This package provides routines to load/save and manipulate images using Torch's Tensor data structure, changing color spaces, rotate, translate, warp etc.
* graphicsmagick - A wrapper to GraphicsMagick (binary). GraphichsMagick (.org) is a tool to convert images, quite efficiently. This package provides bindings to it. 
* imgraph - An image/graph library for Torch. This package provides routines to construct graphs on images, segment them, build trees out of them, and convert them back to images. 


Videos
------------
* camera - A simple wrapper package to give torch access to a webcam
* ffmpeg - An FFMPEG interface for Torch. A simple abstraction class, that uses ffmpeg to encode/decode videos, and represent them as Tensors, in Torch.
* videograph - A video/graph library for Torch. This package provides routines to construct graphs on videos, segment them, build trees out of them, and convert them back to videos.

Audio
------------
* audio - Audio library for Torch-7. Support audio I/O (Load files) Common audio operations (Short-time Fourier transforms, Spectrograms).
* [lua-sndfile](https://github.com/andresy/lua---sndfile) - An interface to libsndfile 
* [lua-pa](https://github.com/andresy/lua---pa) - Interface to PortAudio library

Natural Language Processing
------------
* nn - Neural language models such as ones defined in [Natural Language Processing (almost) from Scratch](http://arxiv.org/abs/1103.0398) can be implemented using the nn package. nn.LookupTable is useful in this regard.

Sensor Input/Output
------------
* camera - A simple wrapper package to give torch access to a webcam
* [mongoose 9dof](https://github.com/MichaelMathieu/lua---mongoose) - Lua/Torch bindings for the Mongoose 9DoF IMU
* [AR.Drone](https://github.com/MichaelMathieu/lua---ardrone) - AR.Drone bindings for torch by @MichaelMathieu

Distributed Computing / Parallel Processing
------------
* parallel - A package to easily fork processes, for Torch
* thmap - Map jobs onto th nodes.
* threads - A FFI threading system based on SDL2 by Ronan Collobert
* [lua-llthreads](https://github.com/Neopallium/lua-llthreads) - Low-Level threads(pthreads or WIN32 threads) for Lua.
* [MPIT](https://github.com/sixin-zh/mpiT) - MPI for Torch by Sixin Zhang

Alternative REPLs
------------
* trepl - An embedabble, Lua-only REPL for Torch.
* env - Adds pretty printing of tensors/tables and additional path handling to luajit 

Utility libraries
------------
* [torch-dokx](https://github.com/d11/torch-dokx) - An awesome automatic documentation generator for torch7
* argcheck - Advanced function argument checker
* buffer - A buffer object for LuaJIT (to get around LuaJIT limitations)
* class - Simple object-oriented system for Lua, with classes supporting inheritance. 
* curl - An interface to CURL.
* cwrap - Advanced automatic wrapper for C functions
* eex - Torch extras from e-Lab
* fn - Some functional programming tools for Lua and Torch.
* fs - File system toolbox
* inline-c - A package to dynamically build and run C from within Lua. Each function gets wrapped in it own little lua library which is then made accessible to the Lua runtime.
* logroll - A basic logging library for Lua and Torch.
* paths - Paths manipulations
* persist - A persisting table, built on Redis.
* pprint - A pretty print library for Torch and lua
* python - A wrapper to Python
* restclient - A REST Client.
* sys - A system library for Torch
* totem - Alternate torch unit test module
* underscore - Underscore is a utility-belt library for Lua
* utf8 - Basic UTF-8 support.
* util - Random utilities for Lua and Torch.
* xlua - Extra Lua functions. Lua is pretty compact in terms of built-in functionalities: this package extends the table and string libraries, and provide other general purpose tools (progress bar, ...). 
* MobDebug - MobDebug is a remote debugger for the Lua programming language
* OAuth - Lua OAuth, an OAuth client library.
* penlight - Lua utility libraries loosely based on the Python standard libraries
* SocialLua - Library for interfacing with many sites and services
* twitter - A Twitter library for Lua.
* uuid - Generates uuids in pure Lua
* [torch-ipython](https://github.com/d11/torch-ipython) - An ipython kernel for torch
* [net-toolkit](https://github.com/Atcold/net-toolkit) - This package allows to save and retrive to/from disk a lighter version of neural networks that is being trained by clearing out their gradients and states.

Interfaces to third-party libraries
------------
* sdl2 - A FFI interface to SDL2
* sundown - A FFI interface to the Markdown implementation of the Sundown library
* fftw3 - A FFI interface to FFTW3
* cairo - A FFI interface to Cairo Graphics
* LUSE - Lua bindings for the Filesystems in Userspace (FUSE) library
* lzmq-ffi - Lua bindings to ZeroMQ
* Readline - Interface to the readline library
* LuaSec - A binding for OpenSSL library to provide TLS/SSL communication over LuaSocket.
* [PLPlot-FFI](https://github.com/sergomezcol/plplot-ffi) - LuaJIT wrapper for PLplot

Asynchronous paradigm - (like nodejs)
------------
* async - An async framework for Torch (based on LibUV)
* redis-async - A redis client built off the torch/lua async framework
* redis-queue - A redis queue framework using async redis

Networking
----------
* LuaSocket - Network support for the Lua language
* nixio - System, Networking and I/O library for Lua

Security
--------
* LuaSec - A binding for OpenSSL library to provide TLS/SSL communication over LuaSocket.

CUDA
------------
* cunn - Torch CUDA Neural Network Implementation
* cutorch - Torch CUDA Implementation

OpenCL
------------
* https://github.com/jonathantompson/jtorch

Miscellaneous
-------------
Packages which I didn't know where to put
* [re](https://github.com/rlowrance/re) - Real Estate software 


Creating your own package
=========================
You can quickly fork off of this example package:

* https://github.com/soumith/examplepackage.torch

Debuggers / Visual Debuggers / IDEs
===================================
* [ZeroBrane Studio](http://studio.zerobrane.com) -Provides a great IDE and visual debugging.
* [zbs-torch](https://github.com/soumith/zbs-torch) - Use this to debug qlua based programs 

GPU Support
===========
CUDA Support, CUDA examples
--------------------------------
* Torch: CUDA is supported by installing the package __cutorch__ . You get an additional tensor type torch.CudaTensor (just like torch.FloatTensor). CUDA double precision is not supported.

* NN: Install the package __cunn__

__Caveats__: __SpatialConvolution__ is really slow on CUDA, instead use __SpatialConvolutionCUDA__ and __SpatialMaxPoolingCUDA__ modules. 
These two modules only support batch mode, and take samples of format ( Depth x Height x Width x Batch ). So you might have to wrap the modules with __nn.Transpose__.

Example of using SpatialConvolutionCUDA is here: 

https://github.com/soumith/galaxyzoo

OpenCL support, OpenCL examples
--------------------------------
There is barely any OpenCL support.

The only known public OpenCL code is by @jonathantompson over here:

https://github.com/jonathantompson/jtorch

Gotchas
=======
LuaJIT limitations
------
2GB and addressing limit

http://hacksoflife.blogspot.com/2012/12/integrating-luajit-with-x-plane-64-bit.html