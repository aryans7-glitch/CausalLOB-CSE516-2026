# Data

## Principal Dataset
FI-2010 Limit Order Book Dataset — NoAuction Decimal Precision.

## Official Source
https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

## Kaggle Mirror
https://www.kaggle.com/datasets/ulfricirons/fi-2010

## Dataset Used

The project reconstructs:
- Day 1 from `Train_Dst_NoAuction_DecPre_CF_1.txt`
- Days 2–10 from `Test_Dst_NoAuction_DecPre_CF_1.txt` through `CF_9.txt`

The canonical dataset contains:
- 394,337 observations
- 40 raw LOB variables
- 5 future-direction labels

Only raw LOB rows 1–40 and labels 145–149 are used in the primary pipeline.

Publisher-provided engineered features are excluded from the primary causal representation.

## Important Note
The raw FI-2010 dataset is not uploaded to GitHub because it is large. The Colab notebook downloads it automatically if the saved checkpoint is unavailable.
