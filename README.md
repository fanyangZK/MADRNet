# MADRNet
Morphology-Aware Dual-path Reversible Net for Sperm Recognition


![network](https://github.com/user-attachments/assets/4f33ed75-b798-4908-b266-f290b1953075)



Sperm morphology analysis has important scientific research value in the clinical diagnosis of male infertility. Although traditional microscopy techniques have been widely used, their inherent methodological limitations need to be taken seriously. The current technology mainly relies on subjective visual assessment by physicians, which may lead to inter observer differences in diagnostic results. Meanwhile, the time-consuming and laborious manual evaluation process, as well as the inconsistency of diagnostic criteria, may affect the accuracy of the final diagnosis. In response to the above challenges, we propose an innovative morphology guided dual path deep network **MADRNet**. The network adopts the dual attention mechanism of parallel space and channel, and embeds the anatomical constraints of sperm acrosome in channel attention. Based on the WHO sperm morphology standard, we designed a dynamic constraint loss function to enhance compliance with international standards. To improve computational efficiency, the network adopts a reversible architecture design, successfully reducing GPU memory consumption by \textbf{24.3\%}. Experiments on the HuSHeM dataset showed that the model achieved an accuracy of \textbf{96.3\%} and an F1 score of \textbf{96.8\%}, while maintaining a real-time processing speed of \textbf{32}ms per image, providing a precise and efficient solution for clinical sperm screening.

## File structure

```
root/
├── data # datasets
│   └── HuSHem # Human Sperm Head Morphology datasets
├── model # MADRNET config file
│   ├── MADRNet.py
│   ├── DualAttention.py
│   └── ReversibleBlock.py
|   └── HybirdLoss.py
├── utils 
│   ├── calculate_metrics.py
│   └── clean_hidden_fils.py   └── 
│   └── Config.py
│   └── ImageDataset.py
│   └── MetricLogger.py
│   └── test_epoch.py
│   └── train_epoch.py
│   └── transform_config.py
├── main.py
├── .gitattributes
├── pre.pth
└── README.md
```

## Run Code

```
python main.py
```

## Notes
### 1.Install Dependencies
```
pip install -r requirements.txt
```

You might want to pin specific versions if you need exact reproducibility. For production use, consider using:
torch==2.0.1+cu118 (with CUDA version)
torchvision==0.15.2+cu118
etc., depending on your system configuration.
### 2.Runtime Environment
Requires Python 3.9+, CUDA 11.8, and PyTorch 2.0+
### 3.Configuration Parameters
Modify Parameters in the Config Class of main.py as Needed
