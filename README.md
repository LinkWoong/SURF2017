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

Our goal is to implement the neural style transfer by using [cycleGAN](https://arxiv.org/abs/1705.09966). 

## 2. Framework  

Despite so many existing and well-performed deep learning frameworks (like caffe, chainer etc), our group chooses **Tensorflow** for its reliability and adaptability. 

### Edge detection  

Edge detection based on **Kears** deep learning framework has been implemented, and test image is  
<br>
<img src="https://github.com/LinkWoong/SURF2017/blob/master/Keras-Implemented-Edge-Detection/test.jpg" height=300 width=50% alt="Input test image"> <img src="https://github.com/LinkWoong/SURF2017/blob/master/Keras-Implemented-Edge-Detection/result2.jpg" height=300 width=50% alt="Output test image">

<br>
The performance is not bad, and for non-anime photo the output is  
<br>
<img src="https://github.com/LinkWoong/SURF2017/blob/master/Keras-Implemented-Edge-Detection/result3.jpg" height=300 width=50% alt="Output test image"> <img src="https://github.com/LinkWoong/SURF2017/blob/master/Keras-Implemented-Edge-Detection/result4.jpg" height=300 width=50% alt="Output test image">

<br>

