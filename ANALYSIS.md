# Results and Analysis

In this section, I analyze and interpret the results obtained from my three CNN models across different tasks and explain my reasoning on which model to choose and how to optimize it in each case. At the end, I will include how I would continue this project in the future. 

## Table of Contents
- [Overall model performance ](#overall-model-performance)
- [Dataset distribution](#dataset-distribution)
- [Contributing](#contributing)

### What questions are we analysing?

The main goal is to choose the model with the highest accuracy because we want a reliable, optimized model. We have three models, and we compare them and see which performs the best overall. The second question is how well does each model perform towards the two most harmful classes, and how restricted is our choice of model based on this analysis, as we don't want to encounter any false negative? The last question that I asked was, would the decision change on these previous observations with a more user-realistic approach, as all images used were high quality and specially designed for optimal results?

## Overall model performance 

We observe that ResNet and EfficientNet perform much better than my own constructed CNN. The ResNet performs 1.6% better in test accuracy than EfficientNet, meaning if we just look at which model performs the best, we would choose and optimize ResNet. Additionally, we notice that the ResNet is more confident in these decisions than the EfficientNet and my own CNN. Furthermore, we notice that both transfer models (ResNet and EfficientNet) perform superior to my own CNN, showcasing the strength of pretrained models. It took me a while to construct an acceptable CNN, whereas with transfer learning, it took me minutes. With this in mind, if applicable, utilize transfer learning. 

<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Loss</th>
  </tr>
  <tr>
    <td>own CNN</td>
    <td>0.898</td>
    <td>0.407</td>
  </tr>
  <tr>
    <td>ResNet50</td>
    <td>0.967</td>
    <td>0.120</td>
  </tr>
  <tr>
    <td>EfficientNetB0</td>
    <td>0.951</td>
    <td>0.156</td>
  </tr>
</table>

*Chosen model: ResNet50*

### Analysing missclassification of each model 

The next step is to optimize model performance and I investigated the missclassification and inspect how to improve. 

point out  

First, we analyze the performance of each model cross all class. Looking at the Histrogramm 
![Histogramm](images/Histogramm_model_comparision.png)


![Pie Chart](images/plot_misclassification_pie_own.png)
![Pie Chart](images/plot_misclassification_pie_resnet.png)
![Pie Chart](images/plot_misclassification_pie_efficient.png)


## Dataset distribution

![Class Distribution](images/dataset.png)


### Dataset Class Distribution



<!--
![Image Missclassification](images/image_missclassification.png)
-->

![Histogramm](images/Histogramm_model_comparision_aug.png)


![Image augmentation](images/image_difference.png)



![Pie Chart](images/plot_misclassification_pie_aug_own.png)
![Pie Chart](images/plot_misclassification_pie_aug_resnet.png)
![Pie Chart](images/plot_misclassification_pie_aug_efficient.png)


![Pie Chart](images/virus_on_models.png)
![Pie Chart](images/virus_on_models_aug.png)


## Installation
