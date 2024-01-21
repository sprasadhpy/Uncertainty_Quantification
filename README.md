# Uncertainty Quantification
Coursework part of the Course Bayesian DeepLearning (COMP0171)

The primary objective is to explore and implement uncertainty quantification techniques within deep learning models, focusing on the robustness and reliability of predictions.

Secondary Objectives :

1. Implement stochastic gradient Langevin dynamics for sampling from a Bayesian neural network
2. Break down the variance to estimate epistemic and aleatoric uncertainty


## Features : 

**Bayesian Deep Learning:** Application of Bayesian principles to deep learning models for enhanced interpretability.

**Uncertainty Quantification:** Implementation and demonstration of uncertainty quantification in model predictions.

**Confidence Visualization:** Using contour plots to visualize the confidence and uncertainty levels in the model's predictions, especially for classification tasks.

## Log-joint minibatch

![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/b86204c4-d391-4331-ba37-4a6470746d4d)


## Log probability Estimates 

![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/ff30073c-8746-477d-922a-25734ebc5561)


## MAP Estimate 

![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/0480ae27-9674-45f9-ac56-4e3fc2ae6ab8)


## Confidence plot 

![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/e9338ed0-0d6e-4a4b-b328-80b4a07ae1f0)


## Calibration and reliability diagrams

ECE = 0.0396

![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/66c3d32a-7267-4762-86f3-ae50e901f5c1)


## Stochastic gradient Langevin dynamics (SGLD)

The `SGLD_step` function below should take a current set of network parameters $\theta$, and update them as

$$\theta' = \theta + \frac{\epsilon^2}{2} \nabla_\theta \log p(\theta, y | X) + \epsilon z$$

where $\epsilon$ is a learning rate, $X, y$ are a current mini-batch, and $z \sim \mathcal{N}(0, I)$ and has the same dimensionality as $\theta$.


## Cyclic learning rate Scheduler 


![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/1d7adaa8-8795-44e7-a8fb-aadc39c3dd81)



## MCMC Parameters 

![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/cb544cbe-ba13-4c02-8ee1-3a273a771fc8)




## Confidence plot and reliability diagrams for the Bayesian classifier


![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/9a59cc27-4bc3-4b2f-9363-70a24a8bc8e5)



![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/90eedb59-0cf6-48ba-ad06-3299fd9df4f5)


## Epistemic uncertainty

![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/18eaa758-1f0d-4f1a-aa49-6c67dda4905b)


## Aleatoric uncertainty 


![image](https://github.com/sprasadhpy/Uncertainty_Quantification/assets/40602129/5c4437a3-fe43-403f-bf9d-cee5bfc3bbf7)

