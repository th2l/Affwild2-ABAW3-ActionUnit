

:3rd_place_medal: [Affwild2-ABAW3 @ CVPRW 2022](https://ibug.doc.ic.ac.uk/resources/cvpr-2022-3rd-abaw/) - Action Unit Detection - PRL

How to run?
1. Install Docker / Create a python environment using conda or other tools.
2. Instead packages in requirements.txt with `pip install -r requirements.txt`, or manually.
3. Edit config file in conf/AU_baseline.yaml, or create a new config file. To use **wandb logger**, install wandb logger, login to wandb, and edit logger param in config file.
4. Run `python main.py --cfg /path/to-config-file`, e.g. `python main.py --cfg /conf/AU_baseline.yaml`
5. Sample scripts for training and testing in `scripts/` folder.

