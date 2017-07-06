# SURF2017
-----------------
This is a project of XJTLU 2017 Summer Undergraduate Research Fellowship, it aims at designing a generative adversarial network to implement style transfer from a style image to content image. Related literature could be viewed from [Wiki](https://github.com/LinkWoong/SURF2017/wiki)  
## 1. Overview  
-----------------
**Neural Style Transfer** is one of the cutting-edge topic in deep learning field. Given an colored image, like [this](https://github.com/titu1994) proposed, and another image that contains the style desired, they could be combined by using **Neural Style Transfer** and it looks like this.
<br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/Dawn%20Sky.jpg?raw=true" height=300 width=50% alt="dawn sky anime"> <img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/style/starry_night.jpg" height=300 width=49%>

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/Dawn-Sky-Mask.jpg?raw=true" height=300 width=50%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Dawn_Sky_masked.jpg?raw=true" height=300 width=49% alt="dawn sky style transfer anime">

<br>

Our goal is to implement the neural style transfer by using [cycleGAN](https://arxiv.org/abs/1705.09966). At the same time, we also want to take one step further by using [CAN](https://arxiv.org/abs/1706.07068), which could generate image itself after a well-feed training process.

## 2. Framework  

Despite so many existing and well-performed deep learning frameworks (like caffe, chainer etc), our group chooses **Tensorflow** for its reliability and adaptability. 

### Edge detection  

Edge detection based on **Kears** deep learning framework has been implemented, and test image is  
<br>
<img src="https://github.com/LinkWoong/SURF2017/blob/master/Keras-Implemented-Edge-Detection/test.jpg" height=512 width=49% alt="Input test image"> <img src="https://github.com/LinkWoong/SURF2017/blob/master/Keras-Implemented-Edge-Detection/result2.jpg" height=512 width=49% alt="Output test image">

<br>
The performance is not bad, and for non-anime photo the output is  
<br>
<img src="https://github.com/LinkWoong/SURF2017/blob/master/Keras-Implemented-Edge-Detection/result4.jpg" height=600 width=98% alt="Output test image">

<br>
There are more results released by using Keras framework, please see this 【link】（http://stellarcoder.com/surf/anime_test/） created by @DexHunter. The network is trained on Professor Flemming 's workstation with 4 Titan X GPUs, which cost 2 weeks to implement. 
