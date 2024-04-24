# Machine Learning for the Fast Online Supernova Pointing

This repo includes the code for the machine learning model used for the Fast Online Supernova Pointing for the DUNE experiment. 
This includes the code for the following problems:
- **Main Track Identification**: Distinguish between the main electron tracks and everything else.
- **Interaction Classification**: Classify the type of interaction (Charged Current on nuclei from Elastic Scattering on electrons) for the main electron track.
- **ES Tracks Direction Regression**: Regress the direction of the ES track. 

## Code structure

The code is structured as follows:
- `python/`: Contains the shared libraries for all the machine learning problems. This is done to have a common evaluation metric, data loading, etc. across all the problems of the same class.
- `scripts/`: Contains the scripts to run the machine learning models for the different problems.
- `json/`: Contains the configuration files for the different problems.
- `es_tracks_dir_regressor/`: Contains the code specific for the ES Tracks Direction Regression problem.
- `interaction_classifier/`: Contains the code specific for the Interaction Classification problem.
- `mt_identifier/`: Contains the code specific for the Main Track Identification problem.

This way, the code is modular and can be easily extended to include new problems.

## Running the code

To run the code on lxplus, you can source the lgc environment with:

```source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh```

Moreover, you need to locally install [healpy](https://healpy.readthedocs.io/en/latest/), to produce the sky maps, and [hyperopt](http://hyperopt.github.io/hyperopt/), to perform hyperparameter optimization, with:

```pip install healpy hyperopt --prefix=/path/to/your/installation/directory```

Then, you can run the code in the `scripts/` directory:

```source run_*.sh -i <json settings> -o <output directory>```

where `<json settings>` is the configuration file you want to use and `<output directory>` is the directory where the output files will be saved.







