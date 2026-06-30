# Competition Notes

This is a Kaggle code competition.

- Submit a notebook, not a direct CSV upload.
- The submission notebook must create `/kaggle/working/submission.csv`.
- Kaggle runtime limit is 9 hours.
- Kaggle provides 2 T4 GPUs, but every algorithm must still have a runtime guard.

## Data Constraints

Train horizontal columns:
```text
MD,X,Y,Z,ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA,TVT,GR,TVT_input
```

Test horizontal columns:
```text
MD,X,Y,Z,GR,TVT_input
```

Test typewell columns:
```text
TVT,GR
```

Therefore inference cannot use:
```text
ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA
```

Inference also cannot read true `TVT` after Prediction Start. Recursive prediction must write each predicted TVT into the working history before later lag and rolling features are computed.
