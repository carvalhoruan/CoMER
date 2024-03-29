import os

import typer
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything

#seed_everything(7)


def main():
    # generate output latex in result.zip
    #ckp_folder = os.path.join("lightning_logs/hme7k_optuna1_83", f"version_{version}", "checkpoints")
    ckp_folder = "lightning_logs/hme7k_optuna4_83/checkpoints"
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test with fname: {fnames[0]}")

    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year="test", zipfile_path = '../bases/HME100K_sum_sub_7k_bttr_test.zip', eval_batch_size=4)

    model = LitCoMER.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)
