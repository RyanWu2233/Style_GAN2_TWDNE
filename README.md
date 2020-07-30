## Style_GAN2_TWDNE (This Waifu Does not exist)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-green.svg?style=plastic)
![Repo TWDNE](https://img.shields.io/badge/Repository-TWDNE-green.svg?style=plastic)
![Image size 512](https://img.shields.io/badge/Image_size-512x512-green.svg?style=plastic)  
 
![Result_6](./Images/result_6.png)   
----
## Implementation detail  
The virtual Waifu pictures are generate by AI using NVIDIA famous style GAN2 algorithm. The training set is composed of 2500 images generated by TWDNE website.
Resolution of each image is 512 x 512. 

----
## Reference  
> Style GAN2 paper: https://arxiv.org/abs/1912.04958  
> TWDNE website: https://www.thiswaifudoesnotexist.net/  
> Style GAN2 for HD fake human face: https://github.com/RyanWu2233/Style_GAN2_FFHQ  
> NVIDIA official source code: https://github.com/NVlabs/stylegan2  

----
## Training progress    
![Training](./Images/generation.gif)   

----
## More example  
![Result_8a](./Images/result_8a.jpg)    

![Gif](./Images/TWDNE_interpolator.gif)    

----
## Usage:  
Execute or include file named `SG2_main.pu`. Then execute following instruction:  
> `model = GAN()` Create style GAN2 object  
> `model.train(restart=False)` Train model
> `model.predict(24)` Generate fake image with 4 rows and 6 cols  
> `model.predict(6)` Generate fake image with 2 rows and 3 cols  
> `model.predict(2)` Generate fake image with 1 rows and 2 cols  
> `model.save_weights()` Save model  
> `mdoel.load_weights()` Load model  

----



