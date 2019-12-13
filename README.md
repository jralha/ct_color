# ct_color

Initial experiments on applying deep learning to rock x-ray tomography colorization.

CNN regression basic models were not good, neural style transfer based on [Gatys et al. (2015)](https://github.com/leongatys/Gatys2015) was promising but took too long to train, specially for a 3D volume.

CycleGan [results](https://github.com/jralha/ct_color_gan) were much better.
