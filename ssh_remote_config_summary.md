Here's a comprehensive summary of your machine configuration for use as prompt context when making code changes:

### **System Information**
- **OS**: Ubuntu 24.04.1 LTS (Linux kernel 6.14.0-28-generic)
- **Architecture**: x86_64
- **Shell**: bash

### **Hardware Specifications**
- **CPU**: AMD EPYC-Milan Processor (32 cores, 32 sockets)
- **Memory**: 115GB total RAM (110GB available)
- **GPU**: NVIDIA A100-SXM4-40GB (40GB VRAM, CUDA 12.4, Driver 550.163.01)
- **Storage**: 484GB total, 108GB available (78% used)

### **Python Environment**
- **Type**: Virtual Environment (.venv)
- **Version**: Python 3.12.3
- **Executable Path**: python

### **Key Installed Packages**
- **Deep Learning**: 
  - PyTorch 2.8.0 + TorchVision 0.23.0
  - CUDA 12.8 support (nvidia-* packages)
  - xformers 0.0.32.post2
- **Computer Vision**: 
  - Detectron2 0.6
  - OpenCV 4.12.0.88
  - Pillow 11.3.0
- **ML/Data Science**: 
  - NumPy 2.2.6, Pandas 2.3.2, SciPy 1.16.1
  - scikit-learn 1.7.1, matplotlib 3.10.5
- **ML Ops**: 
  - MLflow 3.3.1
  - TensorBoard 2.20.0
- **Other**: 
  - Hydra 1.3.2, OmegaConf 2.3.0
  - timm 1.0.19, YACS 0.1.8

### **Project Context**
- **Framework**: DiffusionDet (Diffusion Model for Object Detection)
- **Repository**: https://github.com/thomasgegout/DiffusionDet.git
- **Branch**: my-main
- **Base Framework**: Detectron2 with custom diffusion-based detection head
- **Datasets**: COCO, LVIS, PubTables-1M, FinTabNet
- **Current Working Directory**: DiffusionDet

### **Environment Characteristics**
- **Virtualization**: Running on KVM hypervisor
- **CUDA Support**: Full CUDA 12.4 with A100 GPU available
- **High Memory**: 115GB RAM suitable for large model training
- **Multi-core**: 32 CPU cores for parallel processing

This is a high-performance machine learning setup optimized for deep learning research, particularly suited for computer vision tasks with diffusion models and object detection workloads.