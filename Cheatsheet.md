A quick page for everything Torch.

* [Newbies](#newbies)
* [Installing and Running Torch](#installing-and-running-torch)
* [Installing Packages](#installing-packages)
* [Tutorials, Demos by Category](#tutorials-demos-by-category)
* [Loading popular datasets](#loading-popular-datasets-)
* [List of Packages by Category](#list-of-packages-by-category)

 [Core Math](#general-math) | [Visualization](#visualization) | **[Utility libraries](#utility-libraries)**
------------- | ------------- | ---------- 
**[Data formats I/O](#data-formats)** | **[Sensor I/O](#sensor-inputoutput)** | **[Databases](#databases)** 
**[Machine Learning](#machine-learning)**  | **[Computer Vision](#computer-vision)** | [NLP](#natural-language-processing)
**[Parallel Processing](#distributed-computing--parallel-processing)** | **[CUDA](#cuda)** | **[OpenCL](#opencl)** |  
**[Images](#images)**  | **[Videos](#videos)** | **[Audio](#audio)**
**[Asynchronous](#asynchronous-paradigm---like-nodejs)** | **[Networking](#networking)** | **[Security](#security)**
[Alternative REPLs](#alternative-repls) | **[Interfaces to third-party libs](#interfaces-to-third-party-libraries)** | [Reinforcement Learning](#reinforcement-learning)
[Miscellaneous](#miscellaneous) |  | 

* [Creating your own package](#creating-your-own-package)
* [Debuggers / Visual Debuggers / IDEs](#debuggers--visual-debuggers--ides)
* [GPU Support](#gpu-support)
* [Gotchas](#gotchas)
* [Shipping proprietary torch scripts (sharing scripts with others without sharing source code](https://github.com/soumith/torch-ship-binaries)

Newbies
=======
1. Read this page end-to-end (especially the Gotchas)
2. Install torch
3. [Go though chapter 1 of the video tutorials](https://github.com/Atcold/torch-Video-Tutorials)
3. [Look at the coding tutorial](#tutorials-demos-by-category)
4. Play around with the interpreter
5. Create your own script
6. [Create your own package](#creating-your-own-package)
7. Contribute! :)

If you are used to numpy, [have a look at this page](https://github.com/torch/torch7/wiki/Torch-for-Numpy-users)
If you are used to Matlab, [have a look at this translation pdf](https://github.com/atamahjoubfar/Torch-for-Matlab-users/blob/master/Torch_for_Matlab_users.pdf)

Installing and Running Torch
============================

Mac OS X and Ubuntu 12+
-----------------------

To make sure that the main install instructions for Mac OS X and Ubuntu 12+ are not replicated, and in a single place, we've put them [here](http://torch.ch/docs/getting-started.html#_).

Gentoo Linux
------------

Some external instructions [here](http://spotofdata.com/installing-torch-gentoo/).

Windows
-------

[Running Torch on Windows](https://github.com/torch/torch7/wiki/Windows)

iOS
---------------
https://github.com/clementfarabet/torch-ios

Android
---------------
https://github.com/soumith/torch-android

EC2 Public AMI
---------------
EC2 AMI with the following pre-installed packages:
* CUDA
* cuDNN
* [torch-distro](https://github.com/soumith/torch-distro) (torch + common packages)
* iTorch
* Anaconda

AMI Id: ami-b36981d8. (Can be launched using g2.2xlarge instance)   
[Launch](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-b36981d8) an instance.

EC2 Bitfusion AMI
---------------
EC2 AMI pre-installed with CUDA, cuDNN, [torch-distro](https://github.com/soumith/torch-distro) (torch + common packages), iTorch, Jupyter, and more. Also includes Bitfusion Boost which lets you combine multiple instances into a single virtual instance for max-performance applications and analysis without any code changes required.

[Launch](https://aws.amazon.com/marketplace/pp/B01B4ZSX5S/ref=_ptnr_referral_docs_torch) an instance from the AWS Marketplace.

Docker Images
-------------
``` docker pull kaixhin/torch ```

CUDA 7.5 version - see [repo](https://github.com/Kaixhin/dockerfiles/tree/master/cuda-torch/cuda_v7.5) for requirements
``` docker pull kaixhin/cuda-torch ```

CUDA 7.0 version - see [repo](https://github.com/Kaixhin/dockerfiles/tree/master/cuda-torch/cuda_v7.0) for requirements
``` docker pull kaixhin/cuda-torch:7.0 ```

CUDA 6.5 version - see [repo](https://github.com/Kaixhin/dockerfiles/tree/master/cuda-torch/cuda_v6.5) for requirements
``` docker pull kaixhin/cuda-torch:6.5 ```

Ubuntu 14.04 + [iTorch notebook](https://github.com/facebook/iTorch) - see docker hub [repo](https://hub.docker.com/r/dhunter/itorch-notebook) for details and usage ``` docker pull dhunter/itorch-notebook ```

Installing Packages
===================
Given a package name, you can install it at your terminal with:
```
luarocks install [packagename]
```

Loading popular datasets 
============================
* [MNIST Loader](https://github.com/andresy/mnist) - by Ronan Collobert
* [CIFAR-10, CIFAR-100](https://github.com/soumith/cifar.torch) - CIFAR loader by Soumith Chintala
* [Imagenet Loader](https://github.com/soumith/imagenetloader.torch)
* [Imagenet multi-GPU training on AlexNet and Overfeat](https://github.com/soumith/imagenet-multiGPU.torch)
* [Another imagenet loading + training by @eladhoffer](https://github.com/eladhoffer/ImageNet-Training)
* [KITTI](https://github.com/Aysegul/torch-KITTI) - KITTI dataset loader by Aysegul Dundar
* [Atari2600](https://github.com/fidlej/aledataset) - Scripts to generate a dataset with static frames from the Arcade Learning Environment
* [torch-INRIA](https://github.com/Atcold/torch-INRIA) - INRIA dataset loader by Alfredo Canziani
* [dataload](https://github.com/Element-Research/dataload) - MNIST, Penn Tree Bank, and generic data loaders.
* [dp](https://github.com/nicholas-leonard/dp) - A pylearn2-like deep learning library with several conveniently wrapped datasets including:
  * MNIST
  * NotMNIST
  * CIFAR-10
  * CIFAR-100
  * Google Street View House Numbers (SVHN)
  * Purdue Face Detection
  * Google Billion Words
  * Penn Tree Bank
  * ImageNet
* [STL10 Loader](https://github.com/eladhoffer/stl10.torch)
* [torch-datasets](https://github.com/rosejn/torch-datasets) - Scripts to load several popular datasets including:
  * BSR 500
  * CIFAR-10
  * COIL
  * Street View House Numbers
  * MNIST
  * NORB

Tutorials, Demos by Category
===================================
Tutorials
---------
* [Deep Learning with Torch, a 60 minute tutorial](https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb)
* [Oxford ML course problems](https://github.com/oxford-cs-ml-2015/)
* [Madbits tutorial](https://web.archive.org/web/20160201031326/http://code.madbits.com/wiki/doku.php)
* [code complementing the tutorials](https://github.com/torch/tutorials)
* [dp Tutorials](http://dp.readthedocs.org/en/latest/#tutorials-and-examples)
  * [Neural Network](http://dp.readthedocs.org/en/latest/neuralnetworktutorial/index.html)
  * [Kaggle Facial Keypoint Detection](http://dp.readthedocs.org/en/latest/facialkeypointstutorial/index.html)
  * [Language Model](http://dp.readthedocs.org/en/latest/languagemodeltutorial/index.html)
* [Torch cheatsheet for Matlab users](https://github.com/atamahjoubfar/Torch-for-Matlab-users)
* [Implementing LSTMs with nngraph](http://apaszke.github.io/lstm-explained.html)
* [Torch Video Tutorials](https://github.com/Atcold/torch-Video-Tutorials): getting started, NN, CNN and RNN theory, implementation and training.

Demos
------
* [Core torch7 demos repository](https://github.com/torch/demos). 
  * loading data
  * tensors
  * linear-regression, logistic-regression
  * face detector (training and detection as separate demos)
  * mst-based-segmenter
  * train-a-digit-classifier
  * train-autoencoder
  * optical flow demo
  * train-on-housenumbers
  * train-on-cifar
  * tracking with deep nets
  * kinect demo
  * filter-bank visualization
  * saliency-networks
* [dp examples](https://github.com/nicholas-leonard/dp/tree/master/examples):
  * [Neural Network](https://github.com/nicholas-leonard/dp/blob/master/examples/neuralnetwork.lua) for Image Classification
  * [Mixture of Experts](https://github.com/nicholas-leonard/dp/blob/master/examples/mixtureofexperts.lua) for Image Classification
  * [Convolution Neural Network](https://github.com/nicholas-leonard/dp/blob/master/examples/convolutionneuralnetwork.lua) for Image Classification
  * [Kaggle Facial Keypoint Detector](https://github.com/nicholas-leonard/dp/blob/master/examples/facialkeypointdetector.lua) using an MLP or a Convolution Neural Network
  * [Neural Network Language Model](https://github.com/nicholas-leonard/dp/blob/master/examples/languagemodel.lua) for Google Billion Words dataset
  * [Recurrent Neural Network Language Model] (https://github.com/nicholas-leonard/dp/blob/master/examples/recurrentlanguagemodel.lua)
* [Imagenet single GPU and multi-GPU demo by Facebook](https://github.com/facebook/fbcunn/tree/master/examples/imagenet)
* [Recurrent Models for Visual Attention](https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua) - for image classification
* [Deep Recurrent Attentive Writer](https://github.com/vivanov879/draw) - for image generation
* [Recurrent Networks with Long Short-term memory (LSTM)](https://github.com/wojzaremba/lstm) - for Language modeling
* [Character Level RNNs with LSTM/GRU](https://github.com/karpathy/char-rnn) - Training character-level language models for Latex, English language, Linux kernel C code etc.
* [Another face detector by @jonathantompson](https://github.com/jonathantompson/geiger_facedetector)
* [Training a Convnet for the Galaxy-Zoo Kaggle challenge(CUDA demo)](https://github.com/soumith/galaxyzoo)
* [Kaggle CIFAR-10 competition](https://github.com/nagadomi/kaggle-cifar10-torch7) Examples of Network-in-Network and GoogleNet Inception modules by @nagadomi
* [Neural Turing Machines](https://github.com/kaishengtai/torch-ntm) - A modified version of NTM with LSTM memory cells
* [Recpool](https://github.com/soumith/recpool) - Reconstructing pooling networks by Jason Rolfe, this branch has some fixes.
* [Music Tagging](https://github.com/mbhenaff/MusicTagging) - Music Tagging scripts for torch7 by Mikael Henaff
* [K-Means feature learning over CIFAR-10: Coates-Single-Layer Networks in Unsupervised Feature Learning](https://github.com/jhjin/kmeans-learning-torch)
* [imagenet loading + training](https://github.com/eladhoffer/ImageNet-Training) - Train imagenet networks, like AlexNet, MattNet etc.
* [Caffe models in Torch](https://github.com/szagoruyko/loadcaffe) - Several state-of-the-art models released to the Caffe PublicZoo can now be used in torch, thanks to @szagoruyko. For non-sequential models like ResNets, try [caffegraph](https://github.com/nhynes/caffegraph).
* [libCCV models in Torch](https://github.com/deltheil/loadccv) - libccv provides state-of-the-art models for computer vision. Load these models in torch and use them naturally.
* [Variational Auto encoders](https://github.com/y0ast/VAE-Torch)
* [Object detection using rcnn and spp](https://github.com/fmassa/object-detection.torch) by fmassa
* [Stacked Hourglass Networks for Human Pose estimation](https://github.com/anewell/pose-hg) - State-of-the-art on the FLIC and MPII benchmarks (2016/3/26).
* [Generating adversarial examples and training a model with adversarial cost](https://github.com/ilarele/torch_examples)

List of Packages by Category
============================
General Math
------------
* [torch](https://github.com/torch/torch7) - The core torch package. Apart from tensor operations, has convolutions, cross-correlations, basic linear algebra operations, eigen values/vectors etc.
* [cephes](http://deepmind.github.io/torch-cephes/) - Cephes mathematical functions library, wrapped for Torch. Provides and wraps the 180+ special mathematical functions from the Cephes mathematical library, developed by Stephen L. Moshier. It is used, among many other places, at the heart of SciPy.
* [graph](https://github.com/torch/graph) - Graph package for Torch
* [randomkit](http://deepmind.github.io/torch-randomkit/) - Numpy's randomkit, wrapped for Torch
* [signal](http://soumith.ch/torch-signal/signal/) - A signal processing toolbox for Torch-7. FFT, DCT, Hilbert, cepstrums, stft

Data formats
------------
* [csvigo](https://github.com/clementfarabet/lua---csv) - A CSV library, for Torch
* [hdf5](https://github.com/d11/torch-hdf5) - Read and write Torch tensor data to and from Hierarchical Data Format files.
* [lua-cjson](http://www.kyne.com.au/~mark/software/lua-cjson.php) - A fast JSON encoding/parsing module
* [dataload](https://github.com/Element-Research/dataload) - Load images from disk using multi-threading, etc. 
* [mattorch](https://github.com/clementfarabet/lua---mattorch) - An interface between Matlab and Torch (This repo is mostly abandoned and not maintained.)
* [matio](https://github.com/soumith/matio-ffi.torch) - Package to load tensors from Matlab .mat files, without having matlab installed on your system. Needs open source libmatio.
* LuaXML - a module that maps between Lua and XML without much ado
* [lua-capnproto](https://github.com/cloudflare/lua-capnproto) Basically like Protocol buffers except MUCH faster 
* LuaZip - Library for reading files inside zip files
* MIDI - Reading, writing and manipulating MIDI data
* [audio](https://github.com/soumith/lua---audio/) - Loads as Tensors, all audio formats supported by libsox (mp3, wav, aac, ogg, flac, etc.)
* [csv2torchdatasets](https://github.com/andreirusu/csv2torch-datasets) - Simple Torch7 tool for converting Kaggle-style csv files to torch-datasets.
* [image](https://github.com/torch/image) - Loads png, jpg, ppm images
* [graphicsmagick](https://github.com/clementfarabet/graphicsmagick) - A full Torch/C interface to GraphicsMagick's Wand API and to imagemagick commandline utility, loads all images thrown its way.
* [ffmpeg](https://github.com/clementfarabet/lua---ffmpeg) - A simple abstraction class, that uses ffmpeg to encode/decode videos from/to Torch Tensors
* [csv2tensor](https://github.com/willkurt/csv2tensor) - Package for loading CSV files containing numeric data directly to Torch Tensors
* [torchx](https://github.com/nicholas-leonard/torchx/blob/master/README.md) - Contains `paths.indexdir` which can be used to efficiently load directories containing thousands of files
* [npy4th](https://github.com/htwaijry/npy4th) - Loads NumPy .npy and .npz
* [torch-dataframe](https://github.com/AlexMili/torch-dataframe) - Loads and manipulates tabular data (e.g. Kaggle-style CSVs) inspired from R's and pandas' data 

Machine Learning
------------
* [nn](https://github.com/torch/nn) - Neural Network package for Torch
* [nngraph](https://github.com/torch/nngraph) - This package provides graphical computation for nn library in Torch7.
* [dp](https://github.com/nicholas-leonard/dp) - A deep learning library designed for streamlining research and development using the Torch7 distribution. 
* [dpnn](https://github.com/nicholas-leonard/dpnn) - deep extensions to nn : `nn.ReverseTable`, `nn.ZipTable`, `nn.Inception`, `nn.Dictionary`, etc.
* [rnn](https://github.com/Element-Research/rnn) - Recurrent Neural Network library for nn
* [nnx](https://github.com/clementfarabet/lua---nnx) - A completely unstable and experimental package that extends Torch's builtin nn library
* [optim](https://github.com/torch/optim) - An optimization library for Torch. SGD, Adagrad, Conjugate-Gradient, LBFGS, RProp and more.
* [hypero](https://github.com/Element-Research/hypero) - Simple distributed hyper-optimization library for torch7
* [nninit](https://github.com/Kaixhin/nninit) - Weight initialisation schemes for nn modules
* [unsup](https://github.com/koraykv/unsup) - A package for unsupervised learning in Torch. Provides modules that are compatible with nn (LinearPsd, ConvPsd, AutoEncoder, ...), and self-contained algorithms (k-means, PCA); see [unsupgpu](https://github.com/viorik/unsupgpu) to run ConvPsd on mini-batches with CUDA.
* [Autoencoders](https://github.com/Kaixhin/Autoencoders) - Selection of autoencoders (sparse, convolutional, denoising, sequence-to-sequence, variational, adversarial)
* [manifold](https://github.com/clementfarabet/manifold) - A package to manipulate manifolds
* [metriclearning](https://github.com/lvdmaaten/metriclearning) - A package for metric learning
* [rbm_toolbox](https://github.com/skaae/rbm_toolbox_lua) - Restricted Boltzmann Machines in torch
* [svm](https://github.com/koraykv/torch-svm) - Torch-SVM library
* [lbfgs](https://github.com/clementfarabet/lbfgs) - FFI Wrapper for liblbfgs
* [vowpalwabbit](https://github.com/clementfarabet/vowpal_wabbit) - An old vowpalwabbit interface to torch.
* [OpenGM](https://github.com/clementfarabet/lua---opengm) - OpenGM is a C++ library for graphical modeling, and inference. The Lua bindings provide a simple way of describing graphs, from Lua, and then optimizing them with OpenGM.
* [sphagetti](https://github.com/MichaelMathieu/lua---spaghetti) - Spaghetti (sparse linear) module for torch7 by @MichaelMathieu
* [LuaSHKit](https://github.com/ocallaco/LuaSHkit) - A lua wrapper around the Locality sensitive hashing library SHKit
* [kernel smoothing](https://github.com/rlowrance/kernel-smoothers) - KNN, kernel-weighted average, local linear regression smoothers
* [component analysis](https://github.com/iassael/component_analysis_torch7) - PCA, Whitened PCA, LDA, LPP, NPP, FastICA.
* [reinforcement learning](https://github.com/Mononofu/reinforcement-learning) - Implementing exercises from Sutton & Barto's textbook Reinforcement Learning: An Introduction
* [warp-ctc](https://github.com/baidu-research/warp-ctc) - A fast parallel implementation of CTC (Connectionist Temporal Classification), useful for performing supervised learning on sequence data, without needing an alignment between input data and labels.
* [torch-autograd](https://github.com/twitter/torch-autograd) - Autograd automatically differentiates native Torch code


Visualization
------------

Mainly provided by three styles:

* [iTorch](https://github.com/facebook/iTorch) - An iPython kernel for torch. Has visualization of images,video, audio and plotting. [Example notebook](http://nbviewer.ipython.org/github/facebook/iTorch/blob/master/iTorch_Demo.ipynb)

or 

* [qtlua](https://github.com/torch/qtlua) - Powerful QT interface to Lua
* [qttorch](https://github.com/torch/qttorch) - QT interface to Torch
* [gnuplot](https://github.com/torch/gnuplot) - Torch interface to Gnuplot
* [iterm.torch](https://github.com/szagoruyko/iterm.torch) - Displays images and other docs right in terminal (OS X only)

or
* [display](https://github.com/szym/display) - A simplified version of gfx.js that is easier to install and use.
* [gfx.js](https://github.com/clementfarabet/gfx.js) - A graphics backend for the browser, with a Torch7 client. Extend this by writing simple html/javascript templates
* [visdom](https://github.com/facebookresearch/visdom) - Like display/gfx.js but with more capabilities, supports Torch

An excellent overview of all plotting packages in torch is given by Florian Strub in this blogpost:
[http://www.lighting-torch.com/2015/08/24/plotting-with-torch7/](http://www.lighting-torch.com/2015/08/24/plotting-with-torch7/)

Computer Vision
------------
* [fex](https://github.com/koraykv/fex) - A package for feature extraction in Torch. Provides SIFT and dSIFT modules. 
* [imgraph](https://github.com/clementfarabet/lua---imgraph) - An image/graph library for Torch. This package provides routines to construct graphs on images, segment them, build trees out of them, and convert them back to images.
* [videograph](https://github.com/clementfarabet/videograph) - A video/graph library for Torch. This package provides routines to construct graphs on videos, segment them, build trees out of them, and convert them back to videos.
* [dp](https://github.com/nicholas-leonard/dp) - A deep learning library for Torch. Includes preprocessing like Zero-Component Analysis whitening, Global Contrast Normalization, Lecun's Local Contrast Normalization and facilities for interfacing your own.
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
* [Overfeat-torch](https://github.com/jhjin/overfeat-torch) - Better Overfeat bindings for Torch by @jhjin
* [similarity-matching-ratio tracker](https://github.com/Aysegul/smr_tracker) - A state-of-the-art tracker by Aysegul Dundar. [More info here](https://sites.google.com/site/ayseguldundar2012/in-the-news/stateofthearttrackersmrsimilarity-matching-ratio)
* [Caffe bindings for Torch](https://github.com/szagoruyko/torch-caffe-binding)
* [torch-opencv](https://github.com/VisionLabs/torch-opencv) - OpenCV 3.0 bindings for LuaJIT+Torch
* [gSLICr-torch](https://github.com/jhjin/gSLICr-torch) - gSLICr superpixel algorithm binding for Torch
* [pcl](https://github.com/Xamla/torch-pcl) - Point Cloud Library (PCL) bindings for Torch
* [gvnn](https://github.com/ankurhanda/gvnn) - Geometric Vision with Neural Networks (includes Spatial Transformer Networks)

Images
------------
* [image](https://github.com/torch/image) - An image library for Torch. This package provides routines to load/save and manipulate images using Torch's Tensor data structure, changing color spaces, rotate, translate, warp etc.
* [graphicsmagick](https://github.com/clementfarabet/graphicsmagick) - A wrapper to GraphicsMagick (binary). GraphichsMagick (.org) is a tool to convert images, quite efficiently. This package provides bindings to it. 
* [imgraph](https://github.com/clementfarabet/lua---imgraph) - An image/graph library for Torch. This package provides routines to construct graphs on images, segment them, build trees out of them, and convert them back to images. 


Videos
------------
* [camera](https://github.com/clementfarabet/lua---camera) - A simple wrapper package to give torch access to a webcam
* [ffmpeg](https://github.com/clementfarabet/lua---ffmpeg) - An FFMPEG interface for Torch. A simple abstraction class, that uses ffmpeg to encode/decode videos, and represent them as Tensors, in Torch.
* [videograph](https://github.com/clementfarabet/videograph) - A video/graph library for Torch. This package provides routines to construct graphs on videos, segment them, build trees out of them, and convert them back to videos.
* [torchvid](https://github.com/anibali/torchvid) - A Torch interface to the FFmpeg libraries for reading video image data. In contrast to lua---ffmpeg, torchvid does not use the ffmpeg executable or write frames to temporary files. This package also supports video transformations with filtergraphs (such as cropping, flipping and resizing).

Audio
------------
* [audio](https://github.com/soumith/lua---audio) - Audio library for Torch-7. Support audio I/O (Load files) Common audio operations (Short-time Fourier transforms, Spectrograms).
* [lua-sndfile](https://github.com/andresy/lua---sndfile) - An interface to libsndfile 
* [lua-pa](https://github.com/andresy/lua---pa) - Interface to PortAudio library

Natural Language Processing
------------
* [nn](https://github.com/torch/nn) - Neural language models such as ones defined in [Natural Language Processing (almost) from Scratch](http://arxiv.org/abs/1103.0398) can be implemented using the nn package. nn.LookupTable is useful in this regard.
* [rnn](https://github.com/Element-Research/rnn) - Recurrent Neural Network library that has a [Recurrent](https://github.com/Element-Research/rnn/blob/master/README.md#rnn.Recurrent) and [LSTM](https://github.com/Element-Research/rnn/blob/master/README.md#rnn.LSTM) that can be used for language modeling.
* [dp](https://github.com/nicholas-leonard/dp) - A deep learning library for Torch. Includes Neural Network [Language Model Tutorial](http://dp.readthedocs.org/en/latest/languagemodeltutorial/index.html) and [Recurrent Neural Network Language Model](https://github.com/nicholas-leonard/dp/blob/master/examples/recurrentlanguagemodel.lua) implementations using the Google Billion Words dataset.
* [senna](https://github.com/torch/senna) - Part-of-speech tagging, Chunking, Name Entity Recognition and Semantic Role Labeling, extremely fast and used to be state-of-the-art (dont know now)
* [word2vec](https://github.com/rotmanmi/word2vec.torch) - Ready to use word2vec embeddings in torch. This repo reads the bin file of word2vec and loads the embeddings to be used. Provides example.
* [GloVe](https://github.com/rotmanmi/glove.torch) - Ready to use GloVe embeddings in torch. This repo reads the text files of GloVe and loads the embeddings to be used. Provides example.
* [torch-word-emb](https://github.com/iamalbert/torch-word-emb) - load Word2Vec and GloVe word embeddings to Torch Tensors, implemented in pure C

Sensor Input/Output
------------
* [camera](https://github.com/clementfarabet/lua---camera) - A simple wrapper package to give torch access to a webcam
* [mongoose 9dof](https://github.com/MichaelMathieu/lua---mongoose) - Lua/Torch bindings for the Mongoose 9DoF IMU
* [AR.Drone](https://github.com/MichaelMathieu/lua---ardrone) - AR.Drone bindings for torch by @MichaelMathieu
* [Arcade Learning Environment](https://github.com/fidlej/alewrap) - A lua wrapper to ALE by Google Deepmind

Distributed Computing / Parallel Processing
------------
* [parallel](https://github.com/clementfarabet/lua---parallel) - A simple yet powerful parallel compute package for Torch. Provides a simple mechanism to dispatch and run Torch/Lua code as independent processes and communicate via ZeroMQ sockets. Processes can be forked locally or on remote machines (via ssh).
* [thmap](https://github.com/clementfarabet/thmap) - Map jobs onto th nodes (built on top of [async](https://github.com/clementfarabet/async))
* [threads](https://github.com/torch/threads-ffi) - An FFI threading system based on SDL2 by Ronan Collobert. More powerful than llthreads, as it allows trivial data sharing between threads.
* [lua-llthreads](https://github.com/Neopallium/lua-llthreads) - Low-Level threads(pthreads or WIN32 threads) for Lua.
* [MPIT](https://github.com/sixin-zh/mpiT) - MPI for Torch by Sixin Zhang
* [lua-mapreduce](https://github.com/pakozm/lua-mapreduce) - A map-reduce framework by Paco Zamora Martínez
* [hypero](https://github.com/Element-Research/hypero) - Simple distributed hyper-optimization library for torch7
* [torch-ipc](https://github.com/twitter/torch-ipc) - A set of primitives for parallel computation in Torch
* [torch-distlearn](https://github.com/twitter/torch-distlearn) - A set of distributed learning algorithms for Torch

Alternative REPLs
------------
* [trepl](https://github.com/torch/trepl) - An embedabble, Lua-only REPL for Torch.
* [env](https://github.com/torch/env) - Adds pretty printing of tensors/tables and additional path handling to luajit 

Utility libraries
------------
##### Utility toolboxes
* [penlight](http://stevedonovan.github.io/Penlight/api/index.html) - Lua utility libraries loosely based on the Python standard libraries
* [moses](https://github.com/Yonaba/Moses) - A more up-to-date alternative to underscore
* [underscore](http://mirven.github.io/underscore.lua/) - Underscore is a utility-belt library for Lua
* [torchx](https://github.com/nicholas-leonard/torchx/blob/master/README.md) - Contains extensions to torch for concatenating tensors, indexing folders, hashing strings and working with nested tables of tensors. 

##### Documentation
* [torch-dokx](https://github.com/d11/torch-dokx) - An awesome automatic documentation generator for torch7
* [dash-docset-torch](https://github.com/ppwwyyxx/dash-docset-torch/) - Dash/Zeal docset for torch and related libraries.

##### File System
* fs - File system toolbox
* [paths](https://github.com/torch/paths) - Paths manipulations

##### Programming helpers
* [argcheck](https://github.com/torch/argcheck) - Advanced function argument checker
* [class](https://github.com/torch/class) - Simple object-oriented system for Lua, with classes supporting inheritance
* [classic](https://github.com/deepmind/classic) - More advanced object-oriented system for Lua
* [cwrap](https://github.com/torch/cwrap) - Advanced automatic wrapper for C functions
* [fn](https://github.com/rosejn/lua-fn) - Some functional programming tools for Lua and Torch.
* [inline-c](https://github.com/clementfarabet/lua---inline-C) - A package to dynamically build and run C from within Lua. Each function gets wrapped in it own little lua library which is then made accessible to the Lua runtime.

##### Printing / Logging / Debugging
* MobDebug - The best debugger for lua. Remote debugger for the Lua programming language
* pprint - A pretty print library for Torch and lua
* logroll - A basic logging library for Lua and Torch.
* xlua - Extra Lua functions. Lua is pretty compact in terms of built-in functionalities: this package extends the table and string libraries, and provide other general purpose tools (progress bar, ...). 

##### Testing
* [totem](https://github.com/akfidjeland/torch-totem) - Alternate torch unit test module

##### Social
* OAuth - Lua OAuth, an OAuth client library.
* SocialLua - Library for interfacing with many sites and services
* twitter - A Twitter library for Lua.

##### Uncategorized
* buffer - A buffer object for LuaJIT (to get around LuaJIT limitations)
* curl - An interface to CURL.
* eex - Torch extras from e-Lab
* python - A wrapper to Python
* [torch-ipython](https://github.com/d11/torch-ipython) - An ipython kernel for torch
* restclient - A REST Client.
* sys - A system library for Torch
* utf8 - Basic UTF-8 support.
* util - Random utilities for Lua and Torch.
* uuid - Generates uuids in pure Lua
* [net-toolkit](https://github.com/Atcold/net-toolkit) - This package allows to save and retrive to/from disk a lighter version of neural networks that is being trained by clearing out their gradients and states.

Databases
---------
* [lmdb.torch](https://github.com/eladhoffer/lmdb.torch) - LMDB Wrapper for Torch
* [ljffi-mongo](https://github.com/Miroff/ljffi-mongo) - MongoDB bindings using Mongo's C driver
* [mongorover](https://github.com/mongodb-labs/mongorover) - MongoDB bindings that easily install into torch7
* lsqlite3 - A binding for Lua to the SQLite3 database library
* LuaSQL-MySQL - Database connectivity for Lua (MySQL driver)
* LuaSQL-Postgres - Database connectivity for Lua (Postgres driver)
* LuaSQL-SQLite3 - Database connectivity for Lua (SQLite3 driver)
* Luchia - Lua API for CouchDB.
* [sqltable](https://zadzmo.org/code/sqltable) - Use sql databases as lua tables, SELECT, INSERT, UPDATE, and DELETE are all handled with metamethods in such a way that no SQL needs to be written in most cases.
* [persist](https://github.com/clementfarabet/persist) - A persisting table, built on Redis.
* redis-async - A redis client built off the torch/lua async framework
* redis-queue - A redis queue framework using async redis
* [luamongo](https://github.com/moai/luamongo) - Lua driver for mongodb (Use v0.2.0 rockspec)
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
* [libmatio](https://github.com/soumith/matio-ffi.torch) - An FFI interface to libmatio and torch wrappers to load tensors as well.
* [ncurses](https://github.com/ocallaco/lua-ncurses) - ncurses wrapper for lua

Asynchronous paradigm - (like nodejs)
------------
* [async](https://github.com/clementfarabet/async) - An async framework for Torch (based on LibUV)
* [redis-async](https://github.com/ocallaco/redis-async) - A redis client built off the torch/lua async framework
* [redis-queue](https://github.com/ocallaco/redis-queue) - A redis queue framework using async redis
* [async-connect](https://github.com/ad2476/async-connect) - A [Connect](http://www.senchalabs.org/connect/) implementation in Lua

Networking
----------
* LuaSocket - Network support for the Lua language
* nixio - System, Networking and I/O library for Lua
* [waffle](https://github.com/benglard/waffle) - Express-like web framework 

Security
--------
* LuaSec - A binding for OpenSSL library to provide TLS/SSL communication over LuaSocket.

CUDA
------------
Using multiple GPUs parallely is supported. Look at [this link](https://github.com/torch/cutorch/issues/42) for more info
* [cutorch](https://github.com/torch/cutorch) - Torch CUDA Implementation
* [cunn](https://github.com/torch/cunn) - Torch CUDA Neural Network Implementation
* [cunnx](https://github.com/nicholas-leonard/cunnx) - Experimental CUDA NN implementations
* [nnbhwd](https://github.com/qassemoquab/nnbhwd) - Convnet implementations with the data-layout BHWD (by Maxime Oquab), very useful for certain applications. 
* [cudnn](https://github.com/soumith/cudnn.torch) - NVIDIA CuDNN Bindings
* [ccn2](https://github.com/soumith/cuda-convnet2.torch) - cuda-convnet2 wrapped for torch
* [FindCUDA](https://github.com/hughperkins/FindCUDA) - Activate incremental builds for cutorch and cunn (experimental)

* [Training an Object Classifier in Torch-7 on multiple GPUs over ImageNet](https://github.com/soumith/imagenet-multiGPU.torch) - Tutorial for training with Torch7 on multiple GPUs.

OpenCL
------------
* [cltorch](https://github.com/hughperkins/cltorch) - Torch OpenCL Implementation
* [clnn](https://github.com/hughperkins/clnn) - Torch OpenCL Neural Network Implementation
* [jtorch](https://github.com/jonathantompson/jtorch)  - Torch cross-platform OpenCL Neural Network Prediction
* [pycltorch](https://github.com/hughperkins/pycltorch) - Wrappers for using cltorch/clnn from Python.

Reinforcement Learning
-------------
* [dpnn](https://github.com/nicholas-leonard/dpnn/blob/master/README.md) - provides modules and criterions that implement the REINFORCE algorithm.
* [rlenvs](https://github.com/Kaixhin/rlenvs) - Reinforcement learning environments
* [rlcore](https://github.com/ludc/rlcore) - Reinforcement learning package (different environments and learning policies) with an interface to openAI Gym
* [Atari](https://github.com/Kaixhin/Atari) - DQN, A3C, and related deep reinforcement learning algorithms

Miscellaneous
-------------
Packages which I didn't know where to put
* [re](https://github.com/rlowrance/re) - Real Estate software 
* [lutorpy](https://github.com/imodpasteur/lutorpy) - A python library for bridging Lua/Torch and Python/Numpy.
* [pytorch](https://github.com/hughperkins/pytorch) - Wrappers for using Lua and Torch from Python.
* [usegpu](https://github.com/lanpa/usegpu) - A convenient script to assign active GPU(s) for current session.

Creating your own package
=========================
You can quickly fork off of this example package:

* https://github.com/soumith/examplepackage.torch

Debuggers / Visual Debuggers / IDEs
===================================
* [Debugging tutorials](https://github.com/Atcold/torch-Developer-Guide#debugging) for *Lua* and *shared object* bindings
* [ZeroBrane Studio](http://studio.zerobrane.com) - Provides a great IDE and visual debugging.
* [LDT](http://www.eclipse.org/ldt/) - An eclipse plugin for Lua.

Some advice from Florian STRUB:
 - Eclipse + plugin. Its debug mode is very powerful with watcher, interactive console, stack-trace and so on. It has nice refactoring features but its coloring is quite poor.
 - zerobrane is the standard Lua IDE, it works well and its debug mode is fine. ~~Yet, you have to use a fork to use torch (zbs-torch).~~ [There is now full integration via a [plugin](http://notebook.kulchenko.com/zerobrane/torch-debugging-with-zerobrane-studio).] To my mind, this IDE is a bit limited. 
 - IntellJ + Lua plugin. The best IDE so far with with nice coloring and excellent refactoring tools. Its debug mode is experimental so it might not work with some of your project.
  
Therefore, I strongly advise you to use Eclipse if you are willing to use advanced debugging features.
http://www.eclipse.org/ldt/

In order to run Torch from Eclipse, Torch's LuaJIT should be added via Windows -> Preferences -> Lua -> Interpreters.

GPU Support
===========
CUDA Support, CUDA examples
--------------------------------
* Torch: CUDA is supported by installing the package __cutorch__ . 
  * Call `require "cutorch"` on a CUDA-capable machine
  * You get an additional tensor type torch.CudaTensor (just like torch.FloatTensor). 
  * CUDA double precision is not supported. 
  * Using **multiple GPUs parallely is supported**. Look at [this link](https://github.com/torch/cutorch/issues/42) for more info

* NN: Install the package __cunn__
  * __Caveats__: __SpatialConvolutionMM__ is the very fast module (on both CPU and GPU), but it takes a little bit of extra memory on the CPU (and a teeny bit extra on the GPU).

  * An alternative is to use NVidia CuDNN, which is very reliable, or cuda-convnet2 bindings, or nnbhwd package
  * [nnbhwd](https://github.com/qassemoquab/nnbhwd) - Convnet implementations with the data-layout BHWD (by Maxime Oquab), very useful for certain applications.
  * [cudnn](https://github.com/soumith/cudnn.torch) - NVIDIA CuDNN Bindings
  * [ccn2](https://github.com/soumith/cuda-convnet2.torch) - cuda-convnet2 ported to torch

NVIDIA Jetson TK1
-----------------
* [Installation and usage instructions for Torch + CuDNN on Jetson TK1](https://github.com/e-lab/torch-toolbox/blob/master/Tutorials/Setup-Torch-cuDNN-on-Jetson-TK1.md)

OpenCL support, OpenCL examples
--------------------------------

* For a fairly complete port of torch to OpenCL, please see [distro-cl](https://github.com/hughperkins/distro-cl)
* For cross-platform GPU prediction, please see [jtorch](https://github.com/jonathantompson/jtorch)

CPU backends
------------

* [nnpack.torch](https://github.com/szagoruyko/nnpack.torch) - very fast convolutions for modern Intel CPUs.

Gotchas
=======
LuaJIT limitations, gotchas and assumptions
-------------------------------------------
Must read! - http://luapower.com/luajit-notes

[2GB and addressing limit](http://hacksoflife.blogspot.com/2012/12/integrating-luajit-with-x-plane-64-bit.html)

Legacy
======

* [Torch3](http://torch.ch/torch3)
* [Torch4](https://github.com/andresy/torch4)
* [Torch5](http://torch5.sourceforge.net)