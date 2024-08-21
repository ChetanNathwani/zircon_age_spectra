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

This can be done by defining an age distribution:
```sh
ages = [0.151, 0.284, 0.293, 0.195, 0.237, 0.21 , 0.367, 0.546, 0.941,
       0.194, 0.422, 0.219, 0.29 , 0.242, 0.319, 0.269, 0.267, 0.138,
       0.217, 0.327, 0.206, 0.263, 0.359, 0.303, 0.449, 0.138, 0.365,
       0.261, 0.142]
```
Then calling the function:
```sh
geochron.filter_older_ages(ages)
```
Here are some examples for three age distributions:

![alt text](https://github.com/ChetanNathwani/zircon_age_spectra/blob/main/readme_figures/readme_filtering.png)

## Online usage
For trying out some of the code without installation, there is an interactive notebook: LINK HERE

## Citation
