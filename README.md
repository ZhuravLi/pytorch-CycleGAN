# pytorch-CycleGAN

This is my final project for DLS MIPT course, the implementation of CycleGAN using PyTorch.
[About CycleGANs](https://hardikbansal.github.io/CycleGANBlog/),
[Link to the original article](https://arxiv.org/abs/1703.10593). 

In this  realization of CycleGAN, there few tricks were applied:
* One-sided label smoothing.
We use targets for real imgs in the discriminator with a value of 0.9.
* We took ReLU as the activation function for generators and LeakyReLU for discriminators.
* During training, a batch size of 1 were used.
* Replay buffer was used to train discriminators. Generated images are added to the replay buffer and sampled from it.

## Visuals
CycleGAN was trained on [monet2photo dataset](https://www.kaggle.com/datasets/balraj98/monet2photo) with 100 epochs and lr=2e-4.
Here some results:

### Monet --> Photo

<div>
<img src="./imgs/A2B/0138_real.png" height="128" width="128">
<img src="./imgs/A2B/0138_fake.png" height="128" width="128">
&nbsp;
<img src="./imgs/A2B/0015_real.png" height="128" width="128">
<img src="./imgs/A2B/0015_fake.png" height="128" width="128">
&nbsp;
<img src="./imgs/A2B/0016_real.png" height="128" width="128">
<img src="./imgs/A2B/0016_fake.png" height="128" width="128">
</div>

<div>
<img src="./imgs/A2B/0127_real.png" height="128" width="128">
<img src="./imgs/A2B/0127_fake.png" height="128" width="128">
&nbsp;
<img src="./imgs/A2B/0148_real.png" height="128" width="128">
<img src="./imgs/A2B/0148_fake.png" height="128" width="128">
&nbsp;
<img src="./imgs/A2B/0029_real.png" height="128" width="128">
<img src="./imgs/A2B/0029_fake.png" height="128" width="128">
</div>

### Photo -> Monet

<div>
<img src="./imgs/B2A/0001_real.png" height="128" width="128">
<img src="./imgs/B2A/0001_fake.png" height="128" width="128">
&nbsp;
<img src="./imgs/B2A/0009_real.png" height="128" width="128">
<img src="./imgs/B2A/0009_fake.png" height="128" width="128">
&nbsp;
<img src="./imgs/B2A/0013_real.png" height="128" width="128">
<img src="./imgs/B2A/0013_fake.png" height="128" width="128">
</div>

<div>
<img src="./imgs/B2A/0102_real.png" height="128" width="128">
<img src="./imgs/B2A/0102_fake.png" height="128" width="128">
&nbsp;
<img src="./imgs/B2A/0199_real.png" height="128" width="128">
<img src="./imgs/B2A/0199_fake.png" height="128" width="128">
&nbsp;
<img src="./imgs/B2A/0024_real.png" height="128" width="128">
<img src="./imgs/B2A/0024_fake.png" height="128" width="128">
</div>

Here the losses:

<img src="./imgs/losses.png">

During the training, discriminator that differs real Monet pics from generated ones is doing his job too well. 
This leads to not good enough performance of generator that creates fake Monet painting.
(Unlike the second generator that creates fake realistic photo. It has done its job pretty well.)

## Installation
Clone the repo

```bash
git clone https://github.com/ZhuravLi/pytorch-CycleGAN
cd pytorch-CycleGAN
```
Then install requirements
```bash
python -m pip install -r requirements.txt
```

## Testing

Setup the dataset. Download it from UC Berkeley's repository:
```bash
./scripts/download_dataset.sh monet2photo
```
Datasets directory looks like this:

    .
    ????????? datasets                   
    |   ????????? <dataset_name>         # monet2photo
    |   |   ????????? trainA             
    |   |   ????????? trainB    
    |   |   ????????? testA        
    |   |   ????????? testB          

Download weights of dataset:
```bash
./scripts/download_model.sh monet2photo
```
Checkpoint directory has the following structure:

    .
    ????????? checkpoints                   
    |   ????????? <dataset_name>_pretrained         # monet2photo_pretrained
    |   |   ????????? gen_A2B.pt                    
    |   |   ????????? gen_B2A.pt  

Finally, testing:

```python
 python -m test --num_test 50
```
Flag `--num_test 50` is responsible for the number of images generated in one direction. You may use `--cuda` for using GPU. You may change paths to generator's weights, if you want to. Use `python -m test --help` to see more details.
Result images are saved at `./output/testing`. After each test, this directory will be cleaned.

## Training

Setup the dataset. Download it from UC Berkeley's repository:
```bash
./scripts/download_dataset.sh monet2photo
```
Datasets directory looks like this:

    .
    ????????? datasets                   
    |   ????????? <dataset_name>         # monet2photo
    |   |   ????????? trainA             
    |   |   ????????? trainB    
    |   |   ????????? testA        
    |   |   ????????? testB          


Finally, training:

```python
 python -m train --start_epoch 0 --num_epochs 10 --cuda --n_cpu 2
```
You may use `--cuda` for using GPU. Use `python -m train --help` to see more details.
When model has done its work, your weights are saved at `checkpoints` directory, that looks like this:

    .
    ????????? checkpoints                   
    |   ????????? <dataset_name>_training         # monet2photo_training
    |   |   ????????? disc_A     
    |   |   |   ????????? disc_A_0001.pt
    |   |   |   :
    |   |   |   ????????? disc_A_0010.pt
    |   |   ????????? disc_B   
    |   |   |   ????????? disc_B_0001.pt
    |   |   |   :
    |   |   |   ????????? disc_B_0010.pt
    |   |   ????????? gen_A2B     
    |   |   |   ????????? gen_A2B_0001.pt
    |   |   |   :
    |   |   |   ????????? gen_A2B_0010.pt          
    |   |   ????????? gen_B2A
    |   |   |   ????????? gen_B2A_0001.pt
    |   |   |   :
    |   |   |   ????????? gen_B2A_0010.pt  
    |   |   ????????? losses
    |   |   |   ????????? losses_0001.pt
    |   |   |   :
    |   |   |   ????????? losses_0010.pt 

All weights after each epoch are saved. After that, you may continue your training using previous weights.

```python
 python -m train --start_epoch 10 --num_epochs 2 --cuda --n_cpu 2
```
Result images are saved at `./output/training` after each epoch.

## License

[Apache](https://choosealicense.com/licenses/apache-2.0/)