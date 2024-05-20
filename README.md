# DriveSceneGen

**[[Paper](https://arxiv.org/abs/2309.14685)] [[Project Page](https://ss47816.github.io/DriveSceneGen/)] [[Code](https://github.com/SS47816/DriveSceneGen)]**

![Alt text](media/cover_white.png)

## DriveSceneGen: Generating Diverse and Realistic Driving Scenarios from Scratch

_Shuo Sun<sup>†</sup>, Zekai Gu<sup>†</sup>, Tianchen Sun<sup>†</sup>, Jiawei Sun, Chengran Yuan, Yuhang Han, Dongen Li, and Marcelo H. Ang Jr._

_Advanced Robotics Centre, National University of Singapore_

_<sup>†</sup>Indicates Equal Contribution_

## Abstract

> Realistic and diverse traffic scenarios in large quantities are crucial for the development and validation of autonomous driving systems. However, owing to numerous difficulties in the data collection process and the reliance on intensive annotations, real-world datasets lack sufficient quantity and diversity to support the increasing demand for data. This work introduces DriveSceneGen, a data-driven driving scenario generation method that learns from the real-world driving dataset and generates entire dynamic driving scenarios from scratch. Experimental results on 5k generated scenarios highlight that DriveSceneGen is able to generate novel driving scenarios that align with real-world data distributions with high fidelity and diversity. To the best of our knowledge, DriveSceneGen is the first method that generates novel driving scenarios involving both static map elements and dynamic traffic participants from scratch. Extensive experiments demonstrate that our two-stage method outperforms existing state-of-the-art map generation methods and trajectory simulation methods on their respective tasks.

## Install

1. Clone this repository

```bash
git clone https://github.com/SS47816/DriveSceneGen.git
cd DriveSceneGen
```

2. Install all Dependencies

```bash
make install
conda activate DriveSceneGen
make pip-install
```

## Usage

### Prepare Training Data

1. Download the official [Waymo Motion Dataset](https://waymo.com/open/licensing/?continue=%2Fopen%2Fdownload%2F) to the `./data/raw` directory

2. Preprocess the downloaded data

    ```bash
    python3 DriveSceneGen/scripts/data_preprocess.py
    ```

3. Plot training data

    ```bash
    python3 DriveSceneGen/scripts/data_rasterization.py
    ```

4. Train the diffusion model

    ```bash
    python3 DriveSceneGen/scripts/train.py
    ```
    
5. Generate scenes using your trained diffusion model

    ```bash
    python3 DriveSceneGen/scripts/generation.py
    ```

6. Vectorize the generated scenes

    ```bash
    python3 DriveSceneGen/scripts/vectorization.py
    ```

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

## License

This repository is licensed under the [Apache License 2.0](https://github.com/SS47816/DriveSceneGen/blob/main/LICENSE)

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
