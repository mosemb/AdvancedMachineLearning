# About the Competition and the MRNet Dataset
The MRNet dataset consists of 1,370 knee MRI exams performed at Stanford University Medical Center.
The dataset contains 1,104 (80.6%) abnormal exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears; labels were obtained through manual extraction from clinical reports. 
The dataset accompanies the publication of the MRNet work [here](https://stanfordmlgroup.github.io/projects/mrnet/).

**Competition details: (from Stanford's website) ** We are hosting a competition to encourage others to develop models for automated interpretation of knee MRs. 
Our test set (called internal validation set in the paper) has its ground truth set using the majority vote of 3 practicing board-certified MSK radiologists (years in practice 6–19 years, average 12 years). 
The MSK radiologists had access to all DICOM series, the original report and clinical history, and follow-up exams during interpretation.

More details from the official website [here](https://stanfordmlgroup.github.io/competitions/mrnet/).

# About Data handling
● Data is loaded as numpy array files.

● The dataset is contains 3 views of MRI Scans
○ Axial
○ Sagittal
○ Coronal

● Each view has exams that have the following interpretations
○ Abnormal
○ ACL
○ Meniscus

● Each exam has multiple scans but not all of them has the same number,
so we used data interpolation in order to make all exams have 24
scans(slices).

● Another approach we took was to only take the 3 middle scans(slices) from each exam.

# Transfer Learning Approach 
● We used ResNet-50 as our base model for feature extraction and added trainable layers. We created 9 models each model perform binary classification on each label.

● Approach Results:

**View Label   Validation Accuracy**

Axial ACL         74.16%

Axial Abnormal    89.16%

Axial Meniscus    70.83%

Coronal ACL       61.67%

Coronal Abnormal  79.16%

Coronal Meniscus  65.83%

Sagittal ACL      75.61%

Sagittal Abnormal 88.33%

Sagittal Meniscus 65.87%

# Original MRNet Paper Approach:

● We used VGG-16 as our feature extractor. Data is fed into the VGG-16 base model then it is passed to a global average pooling layer. The output of GAP layer is passed to a max pooling layer then it is passed to a flatten layer that has after it a dense layer then a prediction layer that has a sigmoid neuron.

● Approach Results:

**View Label Validation Accuracy**

Axial ACL 75%

Axial Abnormal 85%

Axial Meniscus 70.83%

Coronal ACL 66.67%

Coronal Abnormal 85.83%

Coronal Meniscus 68.83%

Sagittal ACL 70.83%

Sagittal Abnormal 83.33%

Sagittal Meniscus 73.33%

# Lastly: The Ensemble Learning: 
● We made ensemble learning using Weighted Majority Voting method, using the 9
models of Transfer Learning Approach and 9 model of MRNet Paper Approach.

● Results: 

**Label Accuracy**

Abnormal  89.16%

ACL       80%

Meniscus  71.67%

**Overall reporting accuracy for our model is 80.28 %**

