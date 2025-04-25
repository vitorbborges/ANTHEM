# ANTHEM project

This is the repository for the final project of the course APPLIED STATISTICS on the spring semester of 2025 at the Politecnico di Milano.

This is a partnership between the MOX laboratory, the Prof. Dr. Lara Cavinato, Postdoctoral fellow Alessandra Ragni and students of the APPLIED STATISTICS course.

## How to contribute

1. Clone the repository
2. Make sure you have all the dependencies installed (see the section above)
3. Create a new branch from `develop` (e.g., `feature/manova` or `feature/pca`, please avoid creating non-informative branch names like `feature/1`, `feature/2` or `feature/coleague_name`)
4. Make your changes
5. If you install a new package, run `pip freeze > requirements.txt` to update the requirements file
6. Commit and Push your changes
7. Open a Pull Request to merge your branch into `develop`

## Data Privacy

Please do not upload any sensitive data to this repository. Every contributor is working under a Non-Disclosure Agreement (NDA) and should not share any data outside the project. For this reason, the `data/` folder is included in the `.gitignore` file.

To download the data, please refer to the shared OneDrive folder or ask the project manager for the data files.

We cannot test code using remote GitHub Actions or cloud-based services that might expose the data. All CI/CD pipelines must be run locally. If you want to approve pull requests into the `develop` branch, please use the [act](https://github.com/nektos/act) tool and ensure that the scripts work locally before merging.
