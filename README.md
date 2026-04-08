# Birdshot-WIDE

**A Dataset of Widefield Fundus Images From Patients With Birdshot Chorioretinitis and Matched Control**

[![DOI](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.19474623-blue)](https://doi.org/10.5281/zenodo.19474623)

## Overview

This repository contains the custom code used for data processing, cohort matching, and the benchmark classification experiment described in the associated data descriptor published in *Scientific Data*.

The Birdshot-WIDE dataset comprises **5,042 widefield fundus photographs** from **742 eyes** with birdshot chorioretinitis (BSCR) and **1,310 images** from **742 age- and sex-matched healthy control eyes**, acquired using the Optos Silverstone ultra-widefield scanning laser ophthalmoscope.

## Repository Structure

```
birdshot-wide/
├── data_processing/
│   ├── 01_dicom_extraction.py       # DICOM to JPEG conversion + metadata extraction
│   ├── 02_patient_aggregation.py    # Aggregate image-level metadata to patient level
│   ├── 03_add_cohort_label.py       # Add cohort column (BCR / Control)
│   ├── 04_add_lesion_type.py        # Assign lesion-type labels from sorted folders
│   └── 05_anonymize_metadata.py     # Pseudonymize patient identifiers (SHA-256)
├── matching/
│   └── case_control_matching.py     # Iterative age- and sex-matched case-control pairing
├── benchmark/
│   └── classification.py            # CNN benchmark (ResNet50, VGG16, EfficientNet-B0, DenseNet121)
├── figures/
│   ├── table1_demographics.py       # Generate Table 1 (cohort demographics)
│   └── figure3_lesion_distribution.py  # Lesion-type frequency plot
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline

The scripts are numbered in the order they should be executed:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `data_processing/01_dicom_extraction.py` | Extract DICOM files from PACS, convert to JPEG, and generate image-level metadata |
| 2 | `data_processing/02_patient_aggregation.py` | Aggregate metadata to one row per patient with image counts per eye |
| 3 | `matching/case_control_matching.py` | Match BSCR patients to controls by sex and age (±10 years) |
| 4 | `data_processing/03_add_cohort_label.py` | Add the `cohort` column to control metadata |
| 5 | `data_processing/04_add_lesion_type.py` | Assign dominant lesion-type labels based on manual sorting |
| 6 | `data_processing/05_anonymize_metadata.py` | Generate pseudonymized patient IDs and produce the final `metadata.xlsx` |
| 7 | `benchmark/classification.py` | Train and evaluate CNN models for BSCR vs. Control classification |

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision
- pydicom
- pandas, openpyxl
- scikit-learn
- Pillow
- matplotlib, seaborn
- [dcmtk](https://dicom.offis.de/dcmtk.php.en) (system package for `dcmj2pnm`)

## Dataset Access

The dataset is available under controlled access on Zenodo:  
**https://doi.org/10.5281/zenodo.19474623**

Prospective users must submit an access request through the Zenodo platform and agree to the Data Usage Agreement (DUA), publicly visible on the repository page.

## Annotation Tool

Lesion annotations were created using [FundusTracker](https://github.com/thomasb6/FundusTracker), a vector-based annotation tool for widefield fundus images.

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{foulonneau2026birdshot,
  title={A Dataset of Widefield Fundus Images From Patients With Birdshot Chorioretinitis and Matched Control},
  author={Foulonneau, Thomas and Memmi, Caroline and Monnet, Dominique and Br{\'e}zin, Antoine P. and Vienne-Jumeau, Ali{\'e}nor},
  journal={Scientific Data},
  year={2026},
  doi={10.5281/zenodo.19474623}
}
```

## License

This code is released under the [MIT License](LICENSE).
