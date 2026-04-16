# Ensemble-Distillation-Based-Adversarial-Training-for-Robust-CIFAR-10-Classification-
A PyTorch framework for robust CIFAR-10 classification, combining ensemble knowledge distillation and adversarial training to defend against FGSM/PGD attacks.
# 🛡️ EDAAT (Ensemble Distillation-Based Adversarial Training)

**Robust CIFAR-10 Classification Pipeline (Research from Dec-2025)**

- GitHub Repository: https://github.com/PranavAshtankar/Ensemble-Distillation-Based-Adversarial-Training-for-Robust-CIFAR-10-Classification-

---

## 🚀 Project Overview

EDAAT is a PyTorch-based, defense-aware adversarial training framework designed to improve the robustness of deep learning models against adversarial perturbations. By combining adversarial learning with knowledge distillation, it achieves strong defensive capabilities while minimizing the "adversarial tax" on clean data.

---

##  Problem Statement

- **Adversarial Vulnerability:** Standard convolutional neural networks are highly susceptible to microscopic adversarial perturbations (like FGSM and PGD).
- **The "Adversarial Tax":** Traditional adversarial training often drastically degrades the model's standard accuracy on clean, unperturbed data.
- **Computation Overhead:** Generating strong adversarial examples for every training step is resource-intensive.
- **Evaluation Blindspots:** Relying solely on basic accuracy fails to capture a model's true worst-case robustness in deployment.

---

##  Solution

A multi-stage training pipeline that acts as a robust defense mechanism. EDAAT utilizes an ensemble of teacher models (e.g., FGSM-trained and Detection models) to guide a student model via knowledge distillation, simultaneously training on both clean and PGD-generated adversarial examples.

---

##  Key Features

- **Defense-Aware Pipeline:** Multi-stage PyTorch architecture seamlessly integrating baseline training and ensemble distillation.
- **High Clean Accuracy:** Maintains a highly competitive ~85–90% clean accuracy on the CIFAR-10 dataset.
- **Optimized for Research:** Features a CPU-optimized, time-balanced training loop with dataset subsetting for rapid prototyping.
- **Multi-Objective Optimization:** Custom loss function balancing clean classification, adversarial defense, and teacher knowledge (Loss = αL_clean + βL_adv + γL_dist).

---

##  Methodology & Evaluation

### 🧪 Phase 1: Baseline & Teacher Training

- **No Defense:** Standard ResNet18 model.
- **Preprocessing:** Defense via average pooling (blurring) to disrupt noise.
- **Gradient Masking:** Injecting Gaussian noise to obscure gradients.
- **Detection & FGSM Adv Training:** These serve as the robust teacher ensemble for the final stage.

###  🛡️ Phase 2: EDAAT Student Training

- **Adversarial Generation:** Real-time generation of PGD attacks.
- **Knowledge Distillation:** Student model learns from the averaged logits of the teacher ensemble.
- **Joint Optimization:** The model backpropagates a weighted sum of clean cross-entropy, adversarial cross-entropy, and KL-Divergence.

###  📊 Advanced Robustness Metrics

- **WCRA (Worst-Case Robust Accuracy):** Evaluates the model against the strongest adversarial perturbation.
- **MRA (Mean Robust Accuracy):** Averages robustness across varying attack intensities.
- **PRG (Performance Robustness Gap):** Measures the arithmetic difference between clean accuracy and robust accuracy.

---

## 🛠️ Technology Stack

- **Core Framework:** PyTorch, Torchvision
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Model Architecture:** ResNet18
- **Environment:** OS, Torch utilities (CPU-optimized setup)

---

## 🎯 Future Enhancements

- **GPU Acceleration:** Scale up PGD steps and utilize full CIFAR-10/CIFAR-100 datasets on CUDA-enabled devices.
- **Stronger Threat Models:** Integrate AutoAttack and Carlini-Wagner (C&W) evaluations.
- **Dynamic Ensemble Weights:** Implement attention mechanisms to dynamically weight teacher models based on batch complexity.
- **Wider Architectures:** Test across WideResNet and Vision Transformer (ViT) backbones.

---

## 🌟 Key Benefits

- **Reduced Adversarial Tax:** Effectively bridges the gap between clean accuracy and adversarial defense.
- **Flexible Architecture:** Modular code design allows for easy swapping of models, datasets, and attack methods.
- **Rapid Prototyping:** Scaled-down dataset parameters allow for quick iterations without requiring heavy compute clusters.
- **Research Ready:** Built on late-2025 paradigms, ideal for academic portfolios and ML security research.

---

## 📄 License

Academic & demonstration use only © 2025

## 👤 Author
[Pranav Ashtankar]
