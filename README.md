<h1 align="center" >
    Dual Residual Dense Network for Image Denoising
</h1>


- First, prepare the datasets into images of sizes 64x64 using the files in the data_preprocessing folder.
- Second, train the model using the following code for a noise level of 10.

``` python main.py --NL 10 --BatchReNormalization True --input_shape (64,64,3) ```

- Third and last, test the performance of the network using using the following line of code

``` python main.py --NL 10 --BatchReNormalization True --input_shape (64,64,3) --test_data_name 'Kodak_Test_C_NL30.h5' ```


<hr>

Tthe test datasets (Kodak and BSDS300) for noise levels 10, 30, and 50 can be downloaded from [here](https://drive.google.com/drive/folders/15YoIPAp4PGn-GxyEMIZCqKrHp7udr8ff?usp=sharing).
