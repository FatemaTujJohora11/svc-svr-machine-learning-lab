# SVC and SVR Machine Learning Lab

**Applied Machine Learning and Neural Networks | Academic Laboratory Project**

**Institution:** Memorial University of Newfoundland, Canada  
**Course:** ENGI 9808 — Applied Machine Learning and Neural Networks

## About the Project

I completed this laboratory project as part of my Applied Machine Learning and Neural Networks coursework at Memorial University of Newfoundland.

The lab involved two machine learning experiments: classifying utensil images using Support Vector Classification (SVC) and predicting used-car selling prices using Support Vector Regression (SVR).

My main focus was to understand how different kernels and hyperparameter settings affect model performance and how validation techniques help compare models.

## Part 1 — Support Vector Classification (SVC)

### Utensil Image Classification

In the first experiment, I used SVC to classify images into three categories: forks, knives, and spoons.

The dataset contained 180 images, with 60 images for each category. I resized the images to 64 × 64 pixels and used Histogram of Oriented Gradients (HOG) to extract image features.

I compared 50 model configurations across four kernels: Linear, Polynomial, RBF, and Sigmoid.

To evaluate the models, I used random holdout validation, K-Fold Cross-Validation, and Repeated K-Fold Cross-Validation.

### SVC Results

| Evaluation | Result |
|---|---|
| Total Images | 180 |
| Classes | Fork, Knife, Spoon |
| Best Repeated CV Accuracy | 82.36% |
| Final Test Accuracy | 77.78% |
| Correct Test Predictions | 21 out of 27 |
| Best Repeated CV Kernel | Polynomial |

The Polynomial kernel performed best during repeated cross-validation, while the RBF kernel produced the highest accuracy in random holdout validation.

Both selected models achieved 77.78% accuracy on the held-out test set.

Although the results did not reach the assigned 90% accuracy target, the experiment helped me understand model selection, overfitting, and the importance of evaluating models on unseen data.

## Part 2 — Support Vector Regression (SVR)

### Used-Car Price Prediction

In the second experiment, I used SVR to predict used-car selling prices based on vehicle characteristics.

The dataset contained 2,000 records with information such as vehicle age, mileage, fuel type, transmission, engine capacity, and other features.

I prepared the data using feature scaling and categorical encoding before training the models.

As with SVC, I compared 50 configurations using Linear, Polynomial, RBF, and Sigmoid kernels.

I evaluated the models using random holdout validation, K-Fold Cross-Validation, and Repeated K-Fold Cross-Validation.

### SVR Results

| Evaluation | Result |
|---|---|
| Total Records | 2,000 |
| Best Performing Kernel | RBF |
| Model 12 Test RMSE | 86,004.41 |
| Model 13 Test RMSE | 86,001.47 |
| Final Test R² | 0.8733 |
| Assigned RMSE Target | 80,000 or lower |

The RBF kernel produced the lowest validation errors among the configurations tested.

The final selected models achieved an R² score of 0.8733, but their RMSE values remained above the assigned target.

This experiment helped me understand how regression models are evaluated and why comparing training, validation, and test results is important.

## Tools and Technologies

- Python
- scikit-learn
- pandas
- NumPy
- Support Vector Machines
- Histogram of Oriented Gradients (HOG)
- Feature Scaling and Encoding
- Cross-Validation
- Model Evaluation

## What I Learned

This lab gave me practical experience in applying machine learning algorithms to both classification and regression problems.

One of my main takeaways was that a model performing well on training data does not necessarily perform equally well on unseen data.

I also learned how different kernels and hyperparameter configurations can lead to different results, even when working with the same dataset.

The experiments helped me develop a better understanding of cross-validation, performance metrics, model comparison, and the limitations of machine learning models.

## Project Limitations

Neither experiment achieved its assigned performance target.

The SVC experiment used a relatively small image dataset, which limited the conclusions that could be drawn from the test results.

The SVR experiment also showed differences between training and validation performance, and some configurations reached their iteration limits.

These limitations were discussed in the original laboratory report.

## Project Status

**Completed Academic Laboratory Project**

This repository documents the experiments and results from my Applied
