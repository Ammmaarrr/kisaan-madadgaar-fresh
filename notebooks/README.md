# Jupyter Notebooks - Plant Disease Detection

This directory contains comprehensive Jupyter notebooks for the AI Project Upgrade Plan.

## 📚 Notebooks Overview

### 1. **01_Agent_Demo.ipynb** ✅
**Purpose:** Progress Report I - Intelligent Agent & Search Algorithms

**Contents:**
- Treatment database setup and visualization
- A* search algorithm demonstrations
- Intelligent agent decision-making scenarios
- Genetic algorithm initialization and operators
- Performance metrics and analysis

**Run Time:** 5-10 minutes  
**Requirements:** Python 3.7+, agent modules

---

### 2. **02_EDA.ipynb** ✅
**Purpose:** Exploratory Data Analysis

**Contents:**
- Dataset overview and structure
- Class distribution analysis
- Data quality assessment
- Sample visualizations
- Train/Val/Test split analysis
- Data augmentation preview
- Recommendations for model training

**Run Time:** 5-15 minutes  
**Requirements:** preprocessing modules, matplotlib, seaborn

---

### 3. **03_Baseline_Models.ipynb** ✅
**Purpose:** Progress Report II - Baseline Models Training

**Contents:**
- **Model 1:** Random Forest Classifier (Traditional ML)
  - Feature extraction with PCA
  - Training and evaluation
  - Feature importance analysis
  - Expected accuracy: ~77%

- **Model 2:** Simple CNN (Deep Learning)
  - Custom 3-layer architecture
  - Training with early stopping
  - Learning curves visualization
  - Expected accuracy: ~86%

- **Model 3:** Transfer Learning (ResNet18)
  - Two-stage fine-tuning strategy
  - Pre-trained ImageNet weights
  - Best performance metrics
  - Expected accuracy: ~93%

**Run Time:** 
- With actual training: 1-2 hours (GPU) or 6-10 hours (CPU)
- Simulated demo: 10-15 minutes

**Requirements:** PyTorch, torchvision, scikit-learn, models modules

---

### 4. **04_Model_Comparison.ipynb** ✅
**Purpose:** Comprehensive Model Analysis and Comparison

**Contents:**
- Performance metrics comparison (accuracy, precision, recall, F1)
- Confusion matrices for all models
- Classification reports
- ROC curves and AUC scores
- Computational efficiency analysis
- Error analysis and confidence evaluation
- Overall model ranking
- Final recommendations

**Run Time:** 10-20 minutes  
**Requirements:** All model results, scikit-learn, matplotlib

---

## 🚀 Quick Start

### Option 1: Run All Notebooks Sequentially

```bash
# 1. Install requirements
pip install -r ../requirements.txt

# 2. Run notebooks in order
jupyter notebook 01_Agent_Demo.ipynb
jupyter notebook 02_EDA.ipynb
jupyter notebook 03_Baseline_Models.ipynb
jupyter notebook 04_Model_Comparison.ipynb
```

### Option 2: Run Individual Notebooks

Each notebook is self-contained and can be run independently for demonstration purposes. They include mock data generation if actual datasets are not available.

```bash
# Run specific notebook
jupyter notebook 02_EDA.ipynb
```

### Option 3: JupyterLab

```bash
# Start JupyterLab
jupyter lab

# Navigate to notebooks/ directory in the sidebar
```

---

## 📋 Prerequisites

### Required Python Packages

```bash
# Core packages
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0

# Machine Learning
scikit-learn>=0.24.0
torch>=1.8.0
torchvision>=0.9.0

# Utilities
jupyter>=1.0.0
pillow>=8.0.0
tqdm>=4.60.0
```

Install all at once:
```bash
pip install -r ../requirements.txt
```

---

## 📊 Expected Outputs

Each notebook generates:

1. **Visualizations** (saved to `../results/`)
   - Charts, plots, graphs
   - Confusion matrices
   - Training curves
   - ROC curves

2. **CSV Reports** (saved to `../results/`)
   - Performance metrics
   - Class distributions
   - Model comparisons

3. **JSON Summaries** (saved to `../results/`)
   - EDA summary
   - Final report
   - Configuration files

4. **Trained Models** (saved to `../saved_models/`)
   - Random Forest: `random_forest_model.pkl`
   - Simple CNN: `simple_cnn_best.pth`
   - ResNet18: `resnet18_best.pth`

---

## 🎯 Usage Scenarios

### For Progress Report I (Week 9)
**Run:** `01_Agent_Demo.ipynb`

This demonstrates:
- Intelligent agent architecture
- A* search algorithm
- Genetic algorithm
- Treatment recommendation system

### For Progress Report II (Week 11)
**Run:** `02_EDA.ipynb` → `03_Baseline_Models.ipynb` → `04_Model_Comparison.ipynb`

This demonstrates:
- Complete data analysis pipeline
- Three baseline models training
- Comprehensive model evaluation
- Final recommendations

---

## 🛠️ Configuration

### Data Path Setup

Before running notebooks, update the data path in each notebook:

```python
# In notebook cells
data_path = '../data/PlantVillage'  # Update this path!
```

### Mock Data Mode

All notebooks support mock data generation for demonstration:
- If data path doesn't exist, mock data is automatically generated
- Mock data maintains realistic distributions
- Perfect for testing and demonstrations

---

## 📁 Directory Structure

```
notebooks/
├── 01_Agent_Demo.ipynb              # Agent & search algorithms
├── 02_EDA.ipynb                      # Exploratory data analysis
├── 03_Baseline_Models.ipynb          # Model training
├── 04_Model_Comparison.ipynb         # Model evaluation
└── README.md                         # This file

../results/
├── eda/                              # EDA outputs
├── training/                         # Training outputs
└── comparison/                       # Comparison outputs

../saved_models/                       # Trained model checkpoints
```

---

## 💡 Tips

1. **Run notebooks in order** for the best experience
2. **Restart kernel** between notebook runs to avoid memory issues
3. **Use GPU** for Model 3 (Transfer Learning) if available
4. **Save frequently** when modifying notebooks
5. **Check output directories** for generated files

---

## ⚠️ Troubleshooting

### Problem: "Module not found"
**Solution:**
```python
# Add parent directory to path (already in notebooks)
import sys
sys.path.append('..')
```

### Problem: "CUDA out of memory"
**Solution:**
- Reduce batch size in config
- Use CPU instead: `device = torch.device('cpu')`
- Close other GPU applications

### Problem: "Data path not found"
**Solution:**
- Update `data_path` variable in notebooks
- Or let notebooks use mock data (automatic)

### Problem: "PyTorch not installed"
**Solution:**
```bash
# CPU version
pip install torch torchvision

# GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 Additional Resources

- **Project Documentation:** `../PROJECT_README.md`
- **Quick Start Guide:** `../QUICK_START.md`
- **Implementation Summary:** `../IMPLEMENTATION_SUMMARY.md`
- **Configuration:** `../config.yaml`

---

## 🎓 Academic Use

These notebooks are designed for academic progress reports:

- **Clear structure** with markdown explanations
- **Visualizations** for presentations
- **Reproducible results** with mock data
- **Comprehensive metrics** for analysis
- **Professional formatting** for reports

---

## ✅ Checklist

Before submission:

- [ ] Run all notebooks successfully
- [ ] Check output directories for generated files
- [ ] Review visualizations (saved as PNG)
- [ ] Verify CSV reports are generated
- [ ] Update data paths if using real dataset
- [ ] Export notebooks as PDF (if required)

---

## 🤝 Contributing

To add new notebooks:
1. Follow existing naming convention
2. Include comprehensive markdown documentation
3. Add mock data support
4. Update this README
5. Test with clean kernel

---

## 📧 Contact

For questions or issues:
- Review documentation in parent directory
- Check troubleshooting section above
- Refer to IMPLEMENTATION_SUMMARY.md

---

**Last Updated:** November 2025  
**Version:** 1.0  
**Status:** ✅ All notebooks complete and tested
