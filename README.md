# imgprocessing_project1

README for Introduction to Image Processing and Computer Vision project 1 - object segmentation.  

We work on [Komatsuna data set.](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/) The data set is composed of pictures of leaves on which we work. The goal is to automate the process of leaf segmentation. We are also given ground truth images using which we can check our results.
  
The project has 2 parts:
1. Segmenting the plants as a whole.  
2. Segmenting each leaf separately.  

Code:  
* Executing <i>ground_truth_init.py</i> produces binary masks of ground truth images with labeled leaves (used for result assesment later on)  
* <i>part1.py</i> produces binary masks for actual input images (not ground truth ones) and compares them with ground truth binary masks (produced by <i>ground_truth_init.py</i>).  
* <i>part2try2.py</i> takes binary masks of plants and tries to separate each leaf.  
* <i>assesment.py</i> compares results for part2 to ground truth images.
