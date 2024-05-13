# DriveSceneGen

**[[Paper](https://arxiv.org/abs/2309.14685)] [[Project Page](https://ss47816.github.io/DriveSceneGen/)] [[Code](https://github.com/SS47816/DriveSceneGen)]**

![Alt text](media/cover_white.png)

## DriveSceneGen: Generating Diverse and Realistic Driving Scenarios from Scratch

_Shuo Sun<sup>†</sup>, Zekai Gu<sup>†</sup>, Tianchen Sun<sup>†</sup>, Jiawei Sun, Chengran Yuan, Yuhang Han, Dongen Li, and Marcelo H. Ang Jr._

_Advanced Robotics Centre, National University of Singapore_

_<sup>†</sup>Indicates Equal Contribution_

## Abstract

> Realistic and diverse traffic scenarios in large quantities are crucial for the development and validation of autonomous driving systems. However, owing to numerous difficulties in the data collection process and the reliance on intensive annotations, real-world datasets lack sufficient quantity and diversity to support the increasing demand for data. This work introduces DriveSceneGen, a data-driven driving scenario generation method that learns from the real-world driving dataset and generates entire dynamic driving scenarios from scratch. DriveSceneGen is able to generate novel driving scenarios that align with real-world data distributions with high fidelity and diversity. Experimental results on 5k generated scenarios highlight the generation quality, diversity, and scalability compared to real-world datasets. To the best of our knowledge, DriveSceneGen is the first method that generates novel driving scenarios involving both static map elements and dynamic traffic participants from scratch.

## BibTeX

If you find our work interesting, please consider citing our paper:

    @misc{sun2023drivescenegen,
        title={DriveSceneGen: Generating Diverse and Realistic Driving Scenarios from Scratch},
        author={Shuo Sun and Zekai Gu and Tianchen Sun and Jiawei Sun and Chengran Yuan and Yuhang Han and Dongen Li and Marcelo H. Ang Jr au2},
        year={2023},
        eprint={2309.14685},
        archivePrefix={arXiv},
        primaryClass={cs.RO}
    }

## Development Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`

### Create a conda env and install dependencies

```bash
make install
conda activate DriveSceneGen
make pip-install
```

Add new dependencies by **manually** editing the `requirements.txt` file

## Code

Code will be made publicly available soon.

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
