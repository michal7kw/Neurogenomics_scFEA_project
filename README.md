# Neurogenomics-Alzheimer-s_mice

Presentation:
https://docs.google.com/presentation/d/19mhx656foUKN7rfJXN9Rs3o84HWM1x0f-PG77bxv3Rc/edit?usp=sharing

Running order:
1. Preprocess Single Cell Data.Rmd
    - load 10X data
    - merge data
    - normalise data
    - annotate cells
    - CREATES: 
        1. counts_AD.csv
        2. counts_stroke.csv
2. data_loading.py
    - load .csv data
    - format data
    - CREATES:
        1. geneExprAD.pkl
        2. geneExprAD_transformed.pkl
        3. geneExprStroke.pkl
        4. geneExprStroke_transformed.pkl
3. data_preparation.py (requires RAM > 32GB)
4. scFEA_with_intermediate_files.py (requires GPU > 32GB)
