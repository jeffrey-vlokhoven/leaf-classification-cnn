# Results and Analysis

In this section, we analyze and interpret the results obtained from my three CNN models across different tasks and explain my reasoning on which model to choose and how to optimize it in each case. At the end, I will include how I would continue this project in the future. 

## Table of Contents
- [Overall model performance ](#overall-model-performance)
- [Dataset distribution](#dataset-distribution)
- [Contributing](#contributing)

### What questions are we analysing?

The main goal is to choose the model with the highest accuracy because we want a reliable, optimized model. We have three models, and we compare them and see which performs the best overall. The second question is how well does each model perform towards the two most harmful classes, and how restricted is our choice of model based on this analysis, as we don't want to encounter any false negative? The last question that I asked was, would the decision change on these previous observations with a more user-realistic approach, as all images used were high quality and specially designed for optimal results?

## Overall model performance 

We observe that ResNet and EfficientNet perform much better than my own constructed CNN. The ResNet performs 1.6% better in test accuracy than EfficientNet, meaning if we just look at which model performs the best, we would choose and optimize ResNet. Additionally, we notice that the ResNet is more confident in these decisions than the EfficientNet and my own CNN, strengthening the decision to choose the ResNet model. Furthermore, we notice that both transfer models (ResNet and EfficientNet) perform superior to my own CNN, showcasing the strength of pretrained models. It took me a while to construct an acceptable CNN, whereas with transfer learning, it took me minutes. With this in mind, if applicable, utilize transfer learning. 

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

The next step would be to optimize model performance until optimal convergence, and to do so, I investigated the misclassification and inspected how to improve. The histogram shows how well each model does across all classes. We notice that all models behave similarly or proportionally across all classes in the sense that the misclassifications are more class-relevant than model-based. There are some outliers, meaning if we have misclassification in one model, we cannot conclude a high probability of misclassification in the other two models. Examples are my own CNN (class 3), EfficientNet (class 5), and ResNet (class 12). With this conclusion, we could optimize multiple classes by optimizing one class, as they might be correlated due to different architectures/approaches coming to the same conclusion. This makes it advantageous and also suggests that optimization may transfer well to other architectures like transformers, as it's not model-based. 

![Histogram](images/Histogramm_model_comparision.png)

Now lets investiges how well each model missclassfied in each class and how we could improve and what improvement might bring the best results. 

#### Model 1: My own CNN

<p float="left">
  <img src="images/plot_misclassification_pie_own.png" alt="Pie Chart" />
  <table>
    <tr><th>Class</th><th>Percentage</th><th>Samples</th></tr>
    <tr><td>Class 7</td><td>28.10%</td><td>59</td></tr>
    <tr><td>Class 10</td><td>16.19%</td><td>34</td></tr>
    <tr><td>Class 6</td><td>13.33%</td><td>28</td></tr>
    <tr><td>Class 11</td><td>11.43%</td><td>24</td></tr>
    <tr><td>Class 12</td><td>10.48%</td><td>22</td></tr>
    <tr><td>Class 3</td><td>9.52%</td><td>20</td></tr>
    <tr><td>Class 9</td><td>5.71%</td><td>12</td></tr>
    <tr><td>Class 8</td><td>1.90%</td><td>4</td></tr>
    <tr><td>Class 0</td><td>0.95%</td><td>2</td></tr>
    <tr><td>Class 1</td><td>0.48%</td><td>1</td></tr>
    <tr><td>Class 14</td><td>0.48%</td><td>1</td></tr>
    <tr><td>Class 5</td><td>0.48%</td><td>1</td></tr>
    <tr><td>Class 13</td><td>0.48%</td><td>1</td></tr>
    <tr><td>Class 4</td><td>0.48%</td><td>1</td></tr>
  </table>
</p>
<div class="container">
  <!-- Image Section -->
  <div>
    <img src="images/plot_misclassification_pie_own.png" alt="Pie Chart">
  </div>

  <!-- Table Section -->
  <div>
    <table>
      <tr><th>Class</th><th>Percentage</th><th>Samples</th></tr>
      <tr><td>Class 7</td><td>28.10%</td><td>59</td></tr>
      <tr><td>Class 10</td><td>16.19%</td><td>34</td></tr>
      <tr><td>Class 6</td><td>13.33%</td><td>28</td></tr>
      <tr><td>Class 11</td><td>11.43%</td><td>24</td></tr>
      <tr><td>Class 12</td><td>10.48%</td><td>22</td></tr>
      <tr><td>Class 3</td><td>9.52%</td><td>20</td></tr>
      <tr><td>Class 9</td><td>5.71%</td><td>12</td></tr>
      <tr><td>Class 8</td><td>1.90%</td><td>4</td></tr>
      <tr><td>Class 0</td><td>0.95%</td><td>2</td></tr>
      <tr><td>Class 1</td><td>0.48%</td><td>1</td></tr>
      <tr><td>Class 14</td><td>0.48%</td><td>1</td></tr>
      <tr><td>Class 5</td><td>0.48%</td><td>1</td></tr>
      <tr><td>Class 13</td><td>0.48%</td><td>1</td></tr>
      <tr><td>Class 4</td><td>0.48%</td><td>1</td></tr>
    </table>
  </div>
</div>



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
