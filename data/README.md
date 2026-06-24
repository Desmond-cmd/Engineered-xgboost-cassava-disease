# Data

## Important
Raw images are NOT stored in this repository due to file size.
Follow the instructions below to get the data.

## Part 1 — Public Dataset (Kaggle)

### Source
Kaggle Cassava Leaf Disease Classification Competition
https://www.kaggle.com/competitions/cassava-leaf-disease-classification

### Download steps
1. Create a free Kaggle account at kaggle.com
2. Go to the competition link above
3. Click Join Competition then accept the rules
4. Click the Data tab
5. Download train_images.zip and train.csv
6. Unzip train_images.zip
7. Place files in this structure:

```
data/
├── train.csv
└── cassava_images/
    ├── 1000015157.jpg
    ├── 1000201771.jpg
    └── ... (21,397 images total)
```

### Class labels in train.csv
| Label | Disease |
|---|---|
| 0 | Cassava Bacterial Blight (CBB) |
| 1 | Cassava Brown Streak Disease (CBSD) |
| 2 | Cassava Green Mottle (CGM) |
| 3 | Cassava Mosaic Disease (CMD) |
| 4 | Healthy |

### Class distribution (imbalanced — justifies Focal Loss)
| Class | Count | Percentage |
|---|---|---|
| CMD (Mosaic) | 13,158 | 61.5% |
| Healthy | 2,577 | 12.1% |
| CGM | 2,386 | 11.2% |
| CBSD | 2,189 | 10.2% |
| CBB | 1,087 | 5.1% |
| Total | 21,397 | 100% |

## Part 2 — Ghanaian Field Dataset

### Source
Collected from smallholder farms in Ashanti and Eastern regions of Ghana
under KNUST CHRPE ethics clearance.

### Collection protocol
- Tool: KoBoToolbox on Android smartphones
- Leaf position: 3rd or 4th leaf from apex
- Distance: 30-50 cm from leaf
- Lighting: natural daylight only
- Labels: confirmed by CSIR-CRI extension officers

### Target sample size
| Class | Minimum | Target |
|---|---|---|
| CMD | 100 | 200 |
| CBSD | 80 | 150 |
| CBB | 60 | 100 |
| CGM | 60 | 100 |
| Healthy | 80 | 150 |
| Total | 380 | 700 |

### Access
Field images available upon request to the research team.
Contact: eboadi@st.knust.edu.gh

## Data dictionary
| Variable | Type | Source | Description |
|---|---|---|---|
| HSV histogram | Continuous (108 values) | Extracted from image | Colour distribution |
| GLCM texture | Continuous (16 values) | Extracted from image | Contrast, correlation, energy, homogeneity |
| Canny edges | Continuous (3 values) | Extracted from image | Edge density, mean, std |
| Disease label | Categorical (0-4) | Annotated | Disease class |
| Region | Categorical | Field metadata | Anonymised to region level |
