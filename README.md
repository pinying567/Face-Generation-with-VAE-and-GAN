# Face Generation with VAE and GAN

**Author: Pin-Ying Wu**

**Table of contents**
- Overview
- Code
- Result Analysis

## Overview
### Task
Synthesizing faces with generative models, including VAE and GAN, with loss functions analysis.  

* <font size=3>**VAE**</font>
    - A variational autoencoder (VAE) is a type of neural network that learns to reproduce its input, and also map data to latent space. it is worth noting that a VAE model can generate samples by sampling from the latent space.

<center>
<!-- ![] (asset/VAE_arch.png)-->
<!--![](https://i.imgur.com/cz8A2Za.png)-->
<img src=asset/VAE_arch.png width=80%><br> 
</center>


* <font size=3>**GAN**</font>
    - A generative adversarial network (GAN) is a deep learning method in which two neural networks (Generator and discriminator) compete with each other to become more accurate in their predictions.

<center>
<!-- ![] (asset/GAN_arch.png)-->
<!--![](https://i.imgur.com/9LqxmZM.png)-->
<img src=asset/GAN_arch.png width=80%>
</center>

### Dataset
**CelebFaces Attributes Dataset (CelebA)**:
Face attributes dataset with 40 attribute annotations. CelebA has large diversities, large quantities, and rich annotations.

<!-- ![] (asset/celebA.png)-->
<!--![](https://i.imgur.com/QJ1zcH2.png)-->
<img src=asset/celebA.png width=100%>


## Code
### Prerequisites
```
pip install -r requirements.txt
```

### Data Preparation
```
bash ./get_dataset.sh
```
The shell script will automatically download the dataset and store the data in a folder called `face_data`.

### Training
1. VAE model
```
bash ./train_VAE.sh
```

2. GAN model
```
bash ./train_GAN.sh
```

### Checkpoints
| VAE | GAN |
|:---:|:---:|
| [VAE_model](https://www.dropbox.com/s/p8a5ti9on1fzq31/VAE.pkl?dl=1)  |  [GAN_model](https://www.dropbox.com/s/24f4nfqthyvh850/GAN.pkl?dl=1)  |

### Evaluation

1. VAE model
```
python3 VAE_main.py --phase test --checkpoint <checkpoint> --out_img <path_to_output_img>
```

2. GAN model
```
python3 GAN_main.py --phase test --checkpoint <checkpoint> --out_img <path_to_test_img_dir>
```

## Result Analysis
### VAE
#### Testing images and their reconstructed images with Mean Squared Error (MSE)
- We randomly choose 10 testing images and get the reconstructed images from our VAE model. The following table is the 10 testing images and their reconstructed results (reconstructed images and MSE).
<!-- ![] (asset/VAE1.png)-->
<!--![](https://i.imgur.com/PpQjPAv.png)-->
<img src=asset/VAE1.png width=100%> <br>

#### Randomly generated images from Normal distribution
- We utilize decoder in our VAE model to randomly generate images by sampling latent vectors from an Normal distribution, and plot 32 random generated images from our model.
- The figures below shows the sampled output of the model with different values of lambda, which is the hyperparameter to weigh the KL divergence loss (KLD) and the reconstruction loss (MSE). The larger the lambda the more KLD contributes to the total loss. When lambda is smaller (e.g. 0.1, 0.5), MSE dominates the training loss. While some of them are very clear faces, some of them are blurred and unreconizable. On the other hand, when lambda is larger (e.g. 10), KLD plays the major role during training and each output image looks like a face. However, the predicted faces are not as clear as the successful example of smaller lambdas (e.g. 0.1, 0.5) and some of the output images are very similar.

<center>
<!-- ![] (asset/VAE2.png)-->
<!--![](https://i.imgur.com/8oajMCf.png)-->
<img src=asset/VAE2.png width=80%> <br>
</center>


#### Experiments of different weights for KLD loss (lambda)
- After several trials, we found that the model with lambda=1 best balanced the two losses, and thus  produces the best result that every output image is recognizable as a face and the model can predict diverse faces.
<!-- ![] (asset/VAE3.png)-->
<!--![](https://i.imgur.com/yQVSrPM.png)-->
<img src=asset/VAE3.png width=100%>

<!-- ![] (asset/VAE4.png)-->
<!--![](https://i.imgur.com/HNMGutR.png)-->
<img src=asset/VAE4.png width=100%> <br>

### GAN
#### Randomly generated images from Normal distribution
- We use the Generator to randomly generate images. Sample 32 noise vectors from Normal distribution and input them into our Generator and plot 32 random images generated from our model.
- When we were training GAN, we found that the hyperparameters are hard to tune, since it is difficult to find the right balance between the generator and the discriminator training. When the discriminator performs very poorly, it is easy for the generator to cheat the discriminator, so the generated images are not real-like. However, if the discriminator performs too well, the gradient to train the generator will be very small, so the generated images are not good as well. After taking references and tuning the hyperparameters for several times, the generated images becomes better and look like faces.

<center>
<!-- ![] (asset/GAN1.png)-->
<!--![](https://i.imgur.com/aXq1it7.png)-->
<img src=asset/GAN1.png width=80%> <br>
</center>

#### Experiments of Different Random Seeds
- We also tried other random seeds to see if there is any difference. As shown in the figure below, the model still produces diverse faces with different random seeds and the image quality remains similar.

<!-- ![] (asset/GAN2.png)-->
<!--![](https://i.imgur.com/fPJ57d4.png)-->
<img src=asset/GAN2.png width=100%> <br>

### Comparison of VAE and GAN
- Compared to the result of VAE, the images generated by GAN are generally clearer. Some of the generated faces from GAN are very impressive that they really look like real images. However, GAN also produces some weird faces. On the contrary, each image generated by VAE are recognizable as a face, but it is blurrier than the result of GAN. We think GAN can generate more real-like images, but also has more noise. The result of GAN can be improved by spending more time tuning the hyperparameters carefully or designing some techniques to make the training of GAN more stable.

<!-- ![] (asset/VAEGAN.png)-->
<!--![](https://i.imgur.com/yNxyNqp.png)-->
<img src=asset/VAEGAN.png width=100%> <br>