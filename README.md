Repository for filtering and comparing zircon age spectra for ID-TIMS datasets

## Prerequisites

POT: python optimal transport
```sh
pip install POT
```
## Installation
1. Open terminal
2. Change the current working directory to the location where you want the cloned directory.
3. Clone the repository
```sh
git clone https://github.com/ChetanNathwani/zircon_age_spectra.git
```
4. You should now see the repository has appeared in your current working directory

## Functions

### Systematically filter "antecrysts" (older tails) in age distributions

This can be done by defining an age distribution, here we use ([Szymanowski et al. (2023)](https://doi.org/10.1016/j.epsl.2023.118408):
```sh
ages = [0.151, 0.284, 0.293, 0.195, 0.237, 0.21 , 0.367, 0.546, 0.941,
       0.194, 0.422, 0.219, 0.29 , 0.242, 0.319, 0.269, 0.267, 0.138,
       0.217, 0.327, 0.206, 0.263, 0.359, 0.303, 0.449, 0.138, 0.365,
       0.261, 0.142]

unc = [0.006, 0.005, 0.005, 0.007, 0.006, 0.005, 0.004, 0.006, 0.005,
       0.005, 0.005, 0.006, 0.005, 0.006, 0.005, 0.005, 0.005, 0.007,
       0.006, 0.005, 0.007, 0.006, 0.01 , 0.005, 0.015, 0.01 , 0.008,
       0.007, 0.006]
```
Then calling the function:
```sh
geochron.filter_older_ages(ages, unc)
```
Here are some examples of ranked age plots for three age distributions where the red bars are non-filtered ages and grey bars are filtered ages:

![alt text](https://github.com/ChetanNathwani/zircon_age_spectra/blob/main/readme_figures/readme_filtering.png)

### Compare the shape of an age distribution to published ID-TIMS age distributions

We first initialise a ```pd.DataFrame()``` containing the results of a principal component analysis of the Wasserstein dissimilairty matrix of the filtered ID-TIMS age distribution compilation:

```sh
geochron.generate_pca_scores()
```
Let's take a look at what that produced:

|    |        PC1 |         PC2 | Type     | Locality             |
|---:|-----------:|------------:|:---------|:---------------------|
|  0 | -0.465769  | -0.00700539 | Plutonic | Adamello             |
|  1 | -0.391456  | -0.328376   | Volcanic | Agua de Dionisio     |
|  2 | -0.667256  | -0.124333   | Porphyry | Bajo de la Alumbrera |
|  3 | -0.464677  | -0.0586558  | Porphyry | Bajo de la Alumbrera |
|  4 | -0.534635  | -0.286713   | Porphyry | Bajo de la Alumbrera |
|  5 | -0.552401  |  0.225343   | Porphyry | Batu Hijau           |
|  6 | -0.109091  |  0.362511   | Porphyry | Batu Hijau           |
|  7 | -0.251048  | -0.294216   | Porphyry | Batu Hijau           |
|  8 |  0.173185  | -0.343716   | Plutonic | Bear Valley          |
|  9 | -0.38985   | -0.30109    | Plutonic | Bear Valley          |
| 10 | -0.343748  | -0.127883   | Plutonic | Bear Valley          |
| 11 | -0.304634  | -0.436756   | Plutonic | Bergell              |
| 12 |  0.980955  | -0.139933   | Plutonic | Bergell              |
| 13 |  0.203133  | -0.362156   | Plutonic | Bergell              |
| 14 | -0.17997   | -0.288732   | Plutonic | Bergell              |
| 15 |  0.349097  | -0.532901   | Plutonic | Bergell              |
| 16 |  0.0796051 | -0.499426   | Plutonic | Bergell              |
| 17 |  0.700678  | -0.325638   | Plutonic | Bergell              |
| 18 | -0.635634  |  0.168698   | Porphyry | Bingham Canyon       |
| 19 | -0.575977  | -0.204604   | Porphyry | Bingham Canyon       |
| 20 |  0.77727   | -0.183307   | Plutonic | Capanne              |
| 21 |  1.97742   |  0.916772   | Plutonic | Capanne              |
| 22 | -0.473491  | -0.181504   | Plutonic | Capanne              |
| 23 |  0.347536  | -0.394673   | Plutonic | Capanne              |
| 24 | -0.464833  | -0.40269    | Plutonic | Capanne              |
| 25 | -0.0190579 | -0.515712   | Plutonic | Capanne              |
| 26 | -0.227219  | -0.432933   | Volcanic | Carpathian-Pannonian |
| 27 | -0.328832  |  1.25728    | Volcanic | Carpathian-Pannonian |
| 28 | -0.543165  |  0.443821   | Volcanic | Carpathian-Pannonian |
| 29 | -0.245996  | -0.191063   | Volcanic | Chegem               |
...

Now we can cast our example Youngest Toba Tuff age distribution into the same PCA space and compare the results with the age distribution compilation:

```sh
geochron.calc_W_PCA(ages_fil, unc_fil) # Use the filtered age distribution which removes one older outlier
```

An example of the results:

![alt text](https://github.com/ChetanNathwani/zircon_age_spectra/blob/main/readme_figures/readme_PCA_W2_plot.png)

## Online usage
For trying out some of the code without installation, there is an interactive notebook: LINK HERE

## Citation
