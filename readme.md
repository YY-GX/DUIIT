# Discriminative Cross-Modal Data Augmentation for Medical Imaging Applications

This is the Pytorch implementaion of the paper:

**Discriminative Cross-Modal Data Augmentation for Medical Imaging Applications**

## Dependencies
- torchvision
- torch
- tensorboard
- dominate
- visdom
- fastai
- tensorboardX

## File Orgnization
```
- generate_part    :Generate fake CT images from other modalities
    - mri2ct.sh    :Train translator (mri -> ct)
    - pet2ct.sh    :Train translator (prt -> ct)
    - mri_generate.sh   :Generate fake CT images from MRI images using trained translator
    - pet_generator.sh  :Generate fake CT images from PET images using trained translator

- task_part    :Combining fake and real images to do prediction
    - mr_ct.sh    :Do prediction using real and fake(mr2ct) CT images
    - pet_ct.sh   :Do prediction using real and fake(pet2ct) CT images

- datasets 
    - real_ct   :real CT images
    - real_mri  :real MRI images
    - real_pet  :real PET images
    - fake_mr2ct    :fake CT images (MRI->CT)
    - fake_pet2ct    :fake CT images (PET->CT)
    - mri
        - trainA    :Train set of source modality images (Here is MRI)
        - trainB    :Train set of target modality images (Here is CT)
        - valA    :Validation set of source modality images (Here is MRI)
        - valB    :Validation set of target modality images (Here is CT)
    - pet
        - trainA    :Train set of source modality images (Here is PET)
        - trainB    :Train set of target modality images (Here is CT)
        - valA    :Validation set of source modality images (Here is PET)
        - valB    :Validation set of target modality images (Here is CT)
```

## Datasets
We use three datasets:
1. MRI dataset: [Link to MRI dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)
2. PET dataset: [Link to PET dataset](https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT)
2. CT dataset: [Link to CT dataset](https://www.kaggle.com/vbookshelf/computed-tomography-ct-images)

## Usage
#### 1. Download datasets to correponding directories.
#### 2. Train translator.
For MRI -> CT:
```
cd generate_part
sh mri2ct.sh
```
For PET -> CT:
```
cd generate_part
sh pet2ct.sh
```
#### 3. Using trained translator to generate fake CT images.
For MRI -> CT:
```
cd generate_part
sh mri_generate.sh
```
For PET -> CT:
```
cd generate_part
sh pet_generate.sh
```
#### 4. Physiological age prediction using real and fake images.
For MRI + CT:
```
cd task_part
sh mri_ct.sh
```
For PET + CT:
```
cd task_part
sh pet_ct.sh
```

## References
CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix