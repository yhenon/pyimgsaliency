# pyimgsaliency
A python toolbox for image saliency calculation.


The following algorithms are currently implemented for calculating saliency maps:
- Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price and Radomír Měch. "Minimum Barrier Salient Object Detection at 80 FPS."
- Saliency Optimization from Robust Background Detection, Wangjiang Zhu, Shuang Liang, Yichen Wei and Jian Sun, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014
- R. Achanta, S. Hemami, F. Estrada and S. Süsstrunk, Frequency-tuned Salient Region Detection, IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2009), pp. 1597 - 1604, 2009

An example of the use of this package can be seen at demo.py

![bird](http://imgur.com/kVLfhwy "Original image")

![bird](http://imgur.com/5Zu7T5V.png "mbd")
![bird](http://imgur.com/SgywutJ.png "rbd")
![bird](http://imgur.com/t8NeAVi.png "ft")


License
Provided under the Apache 2.0 License. Note that there might be additional restrictions on some algorithms. In particular, the authors of "Minimum Barrier Salient Object Detection at 80 FPS" note that their algorithm is patent pending and may not be used in commercial applications.