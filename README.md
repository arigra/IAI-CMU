# IAI–CMU Collaboration

This repository contains code, experiments, and tools developed as part of the collaboration between IAI and CMU (Carnegie Mellon University).  

---

## **Current Components**
- **Pulsed radar simulation** – generates synthetic radar data for experiments.  
- **Object detection in Range–Doppler maps** – using Faster R-CNN.  
- **OOD analysis GUI** – graphical interface for analyzing out-of-distribution detection performance.  
- **Dataset merging** – combining RADDet, CARRADA, and CRUW datasets into one large dataset for use in multiple subprojects.

---

## **Datasets**
- [RADDet](https://github.com/ZhangAoCanada/RADDet?tab=readme-ov-file)  
- [CARRADA](http://download.tsi.telecom-paristech.fr/Carrada/)  
- [CRUW](https://www.cruwdataset.org/download)  

The merged dataset will be uploaded to an **S3 bucket** once it is set up.  

---


## **Getting Started**
```bash
# Clone the repository
git clone https://github.com/USERNAME/IAI-CMU.git
cd IAI-CMU

# Run the GUI
streamlit run src/OOD_GUI/main.py
