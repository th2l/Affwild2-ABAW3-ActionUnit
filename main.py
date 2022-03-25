"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import os
import pathlib

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary, StochasticWeightAveraging, \
    BasePredictionWriter, LearningRateMonitor
from core.callbacks import MultiStageABAW3

from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from core import config, ABAW3Model, ABAW3DataModule
from core.config import cfg
from core.io import pathmgr

from datetime import datetime
from tqdm import tqdm
import wandb
import numpy as np


class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval: str = 'epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Make prediction folder
        predictions_postfix = 0
        while os.path.isdir(os.path.join(self.output_dir, "predictions_{}".format(predictions_postfix))):
            predictions_postfix += 1

        prediction_folder = os.path.join(self.output_dir, "predictions_{}".format(predictions_postfix))

        os.makedirs(prediction_folder, exist_ok=True)

        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        print('Saved file: ', os.path.join(self.output_dir, "predictions.pt"))
        print('Creating txt files...')
        preds = []
        # ytruths = []
        findexes = []
        video_ids = []
        for k in predictions[0]:
            preds.append(k[0])
            # ytruths.append(k[1])
            findexes.append(k[2])
            video_ids += k[3]

        if cfg.TASK == 'AU':
            preds = np.squeeze(1 * (torch.concat(preds).numpy() >= 0.5))
            header_name = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
        elif cfg.TASK == 'VA':
            preds = np.squeeze(torch.concat(preds).numpy().astype(float))
            header_name = ['valence', 'arousal']
        elif cfg.TASK == 'EXPR':
            preds = np.squeeze(torch.concat(preds).numpy())
            header_name = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        else:
            raise ValueError('Do not support write prediction for {} task'.format(cfg.TASK))

        # ytruths = np.squeeze(torch.concat(ytruths).numpy())
        findexes = torch.concat(findexes).numpy()
        video_ids = np.array(video_ids)

        video_id_uq = np.unique(video_ids)
        num_classes = preds.shape[-1]

        for vd in video_id_uq:
            list_row = video_ids == vd
            list_preds = preds[list_row, :, :].reshape(-1, preds.shape[-1])
            list_indexes = findexes[list_row, :].reshape(-1)

            if np.sum(np.diff(list_indexes) < 0):
                print('Please check: {}. Indexes are not consistent'.format(vd))
            # Remove duplicate rows. Because we split sequentially => only padding at the end
            num_frames = len(np.unique(list_indexes))
            write_prediction = list_preds[:num_frames, :]
            if cfg.TASK in ['AU', 'VA']:
                pd.DataFrame(data=write_prediction, columns=header_name).to_csv(
                    '{}/{}.txt'.format(prediction_folder, vd), index=False)
            elif cfg.TASK == 'EXPR':
                with open('{}/{}.txt'.format(prediction_folder, vd), 'w') as fd:
                    fd.write(','.join(header_name) + '\n')
                    fd.write('\n'.join(write_prediction))
            else:
                raise ValueError('Do not support write prediction for {} task'.format(cfg.TASK))


if __name__ == '__main__':
    config.load_cfg_fom_args("ABAW 2022")
    config.assert_and_infer_cfg()
    cfg.freeze()

    pl.seed_everything(cfg.RNG_SEED)

    pathmgr.mkdirs(cfg.OUT_DIR)
    run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.LOGGER == 'wandb' and cfg.TEST_ONLY == 'none':
        cfg_file_dir = pathlib.Path(cfg.OUT_DIR, '{}_{}'.format(cfg.TASK, run_version))
    else:
        cfg_file_dir = pathlib.Path(cfg.OUT_DIR, cfg.TASK, run_version)
    pathmgr.mkdirs(cfg_file_dir)
    cfg_file = config.dump_cfg(cfg_file_dir)

    if cfg.LOGGER == 'wandb' and not cfg.OPTIM.TUNE_LR and cfg.TEST_ONLY == 'none':
        logger = WandbLogger(project='Affwild2-ABAW3', save_dir=cfg.OUT_DIR, name='{}_{}'.format(cfg.TASK, run_version),
                             offline=False)
        output_dir = cfg_file_dir

    else:
        # raise ValueError('Do not implement with {} logger yet.'.format(cfg.LOGGER))
        print('Use TensorBoard logger as default')
        logger = TensorBoardLogger(cfg.OUT_DIR, name=cfg.TASK, version=run_version)
        output_dir = logger.log_dir

    if cfg.TEST.WEIGHTS != '':
        result_dir = '/'.join(cfg.TEST.WEIGHTS.split('/')[:-1])
    else:
        result_dir = ''

    print('Working on Task: ', cfg.TASK)
    print(cfg.MODEL.BACKBONE, ' unfreeze: ', cfg.MODEL.BACKBONE_FREEZE)
    max_epochs = cfg.OPTIM.MAX_EPOCH if cfg.TEST.WEIGHTS == '' else 1

    abaw3_dataset = ABAW3DataModule()
    abaw3_model = ABAW3Model(do_mixup=False, use_aux=cfg.MODEL.USE_AUX)

    fast_dev_run = False
    richProgressBarTheme = RichProgressBarTheme(description="blue", progress_bar="green1",
                                                progress_bar_finished="green1")

    # backbone_finetunne = MultiStageABAW3(unfreeze_temporal_at_epoch=3, temporal_initial_ratio_lr=0.1,
    #                                         should_align=True, initial_denom_lr=10, train_bn=True)
    ckpt_cb = ModelCheckpoint(monitor='val_metric', mode="max", save_top_k=1, save_last=True)
    trainer_callbacks = [ckpt_cb,
                         PredictionWriter(output_dir=output_dir, write_interval='epoch'),
                         LearningRateMonitor(logging_interval=None)
                         ]
    if cfg.LOGGER in ['TensorBoard', 'none'] and not cfg.OPTIM.TUNE_LR:
        trainer_callbacks.append(RichProgressBar(refresh_rate_per_second=1, theme=richProgressBarTheme, leave=True))
        trainer_callbacks.append(RichModelSummary())

    if cfg.OPTIM.USE_SWA:
        swa_callbacks = StochasticWeightAveraging(swa_epoch_start=0.8,
                                                  swa_lrs=cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR,
                                                  annealing_epochs=1)
        trainer_callbacks.append(swa_callbacks)

    trainer = Trainer(gpus=1, fast_dev_run=fast_dev_run, accumulate_grad_batches=cfg.TRAIN.ACCUM_GRAD_BATCHES,
                      max_epochs=max_epochs, deterministic=True, callbacks=trainer_callbacks,
                      enable_model_summary=False,
                      num_sanity_val_steps=0, enable_progress_bar=True, logger=logger,
                      gradient_clip_val=None,
                      limit_train_batches=cfg.TRAIN.LIMIT_TRAIN_BATCHES, limit_val_batches=1.,
                      # limit_train_batches=0.05, limit_val_batches=0.05,
                      precision=32 // (cfg.TRAIN.MIXED_PRECISION + 1),
                      auto_lr_find=cfg.OPTIM.TUNE_LR,  # auto_scale_batch_size=None,
                      )

    if cfg.TEST_ONLY != 'none':
        print('Testing only. Loading checkpoint: ', cfg.TEST_ONLY)
        print(cfg.TRANF.TARGET)
        if not os.path.isfile(cfg.TEST_ONLY):
            raise ValueError('Could not find {}'.format(cfg.TEST_ONLY))
        # Load pretrained weights
        pretrained_state_dict = torch.load(cfg.TEST_ONLY)['state_dict']
        abaw3_model.load_state_dict(pretrained_state_dict, strict=False)

        # Prepare test set
        abaw3_dataset.setup()
        # Re-evaluate validation set
        print('Re-evaluate validation set')
        trainer.test(dataloaders=abaw3_dataset.val_dataloader(), ckpt_path=None, model=abaw3_model)

        # Generate train prediction
        print('Generate train prediction')
        trainer.predict(dataloaders=abaw3_dataset.train_dataloader(shufflex=False), ckpt_path=None, model=abaw3_model)

        # Generate val prediction
        print('Generate val prediction')
        trainer.predict(dataloaders=abaw3_dataset.val_dataloader(), ckpt_path=None, model=abaw3_model)

        # Generate test prediction
        print('Generate test prediction')
        trainer.predict(dataloaders=abaw3_dataset.test_dataloader(), ckpt_path=None, model=abaw3_model)
        print('Testing finished.')

    elif cfg.OPTIM.TUNE_LR:
        print('Auto LR Find')
        trainer.tune(abaw3_model, datamodule=abaw3_dataset, lr_find_kwargs={})
    else:
        #
        trainer.fit(abaw3_model, datamodule=abaw3_dataset)

        if cfg.LOGGER == 'wandb':
            wandb.run.log_code("./core/", include_fn=lambda path: path.endswith(".py") or path.endswith('.yaml'), )

        print('Pass with best val_metric: {}. Generating the prediction ...'.format(ckpt_cb.best_model_score))
        if cfg.OPTIM.USE_SWA:
            print('Evaluating with SWA')
            trainer.test(dataloaders=abaw3_dataset.val_dataloader(), ckpt_path=None, model=abaw3_model)
            trainer.save_checkpoint(ckpt_cb.last_model_path.replace('.ckpt', '_swa.ckpt'))

        trainer.test(dataloaders=abaw3_dataset.val_dataloader(), ckpt_path='best')
        trainer.predict(dataloaders=abaw3_dataset.test_dataloader(), ckpt_path='best')
