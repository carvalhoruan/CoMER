from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.utilities import seed
from optuna.integration import PyTorchLightningPruningCallback

from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER

GPUS = 1

def train_test_COMER_model(params=None, trial=None):
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor='val_ExpRate', save_top_k=1, mode='max', filename='{epoch}-{step}-{val_ExpRate:.4f}')
    early_stopping = EarlyStopping('val_ExpRate', patience=20, mode='max')
    callbacks = [lr_monitor, checkpoint_callback, early_stopping]

    #if trial is not None:
    #    callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_ExpRate"))

    if params is None:
        params = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': callbacks,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 2, 'max_epochs':400,
                                 'save_config_overwrite': True},
                        model={'d_model': 64, 'growth_rate': 24, 'num_layers': 8, 'nhead': 8, 'num_decoder_layers': 1,
                               'dim_feedforward': 256, 'dropout': 0.0,
                               'dc': 8, 'cross_coverage': True, 'self_coverage': True,
                               'beam_size': 10, 'max_len': 200, 'alpha': 1.0, 'early_stopping':False, 'temperature': 1.0,
                               'learning_rate': 0.08, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'train_batch_size': 8, 'eval_batch_size': 2,
                              'num_workers': 5, 'data_augmentation':1, 'scale_aug':True}
                        )
    else:
        params['trainer']['callbacks'] = callbacks
    
    #seed.seed_everything(params['seed_everything'], workers=True)

    model = LitCoMER(
        d_model=params['model']['d_model'],
        growth_rate=params['model']['growth_rate'],
        num_layers=params['model']['num_layers'],
        nhead=params['model']['nhead'],
        num_decoder_layers=params['model']['num_decoder_layers'],
        dim_feedforward=params['model']['dim_feedforward'],
        dropout=params['model']['dropout'],
        dc=params['model']['dc'],
        cross_coverage=params['model']['cross_coverage'],
        self_coverage=params['model']['self_coverage'],
        beam_size=params['model']['beam_size'],
        max_len=params['model']['max_len'],
        alpha=params['model']['alpha'],
        early_stopping=params['model']['early_stopping'],
        temperature=params['model']['temperature'],
        learning_rate=params['model']['learning_rate'],
        patience=params['model']['patience']
        )
    
    dm = CROHMEDatamodule(
        zipfile_path=params['data']['zipfile_path'],
        test_year=params['data']['test_year'],
        train_batch_size=params['data']['train_batch_size'],
        eval_batch_size=params['data']['eval_batch_size'],
        num_workers=params['data']['num_workers'],
        data_augmentation=params['data']['data_augmentation'],
        scale_aug=params['data']['scale_aug']
    )

    trainer = Trainer(
        #seed_everything=params['seed_everything'],
        #deterministic=True,
        checkpoint_callback=params['trainer']['checkpoint_callback'],
        callbacks=callbacks,
        gpus=GPUS,
        check_val_every_n_epoch=params['trainer']['check_val_every_n_epoch'],
        max_epochs=params['trainer']['max_epochs']
    )
    trainer.fit(model, datamodule=dm)

    #return trainer.callback_metrics["val_ExpRate"].item(), trainer.callback_metrics["train_loss"].item()
    return checkpoint_callback.best_model_score.item(), trainer.callback_metrics["train_loss"].item()
    #return trainer.callback_metrics


if __name__ == '__main__':

    #Para rodar na nossa base
    #Ir em vocab.py e alterar dicionário para dictionary_nossa_base.txt
    # zipfile_path = '../bases/Base_soma_subtracao_optuna_bttr.zip'
    # augmentation = 100

    #Para rodar HME7K
    #Ir em vocab.py e alterar dicionário para dictionary_hme7k.txt
    zipfile_path = '../bases/HME100K_sum_sub_7k_bttr.zip'
    augmentation = 0

    params_1_3 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500,
                                 'save_config_overwrite': True},
                        model={'d_model': 128, 'growth_rate': 16, 'num_layers': 16, 'nhead': 2, 'num_decoder_layers': 3,
                               'dim_feedforward': 256, 'dropout': 0.3,
                               'dc': 8, 'cross_coverage': True, 'self_coverage': True,
                               'beam_size': 5, 'max_len': 200, 'alpha': 1.0, 'early_stopping':False, 'temperature': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': zipfile_path, 'test_year': 'test',
                              'train_batch_size': 8, 'eval_batch_size': 4,
                              'num_workers': 5, 'data_augmentation':augmentation, 'scale_aug':False}
                        )
    
    params_2_1 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500,
                                 'save_config_overwrite': True},
                        model={'d_model': 64, 'growth_rate': 16, 'num_layers': 8, 'nhead': 2, 'num_decoder_layers': 1,
                               'dim_feedforward': 256, 'dropout': 0.3,
                               'dc': 8, 'cross_coverage': True, 'self_coverage': True,
                               'beam_size': 5, 'max_len': 200, 'alpha': 1.0, 'early_stopping':False, 'temperature': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': zipfile_path, 'test_year': 'test',
                              'train_batch_size': 8, 'eval_batch_size': 4,
                              'num_workers': 5, 'data_augmentation':augmentation, 'scale_aug':False}
                        )
    
    params_3_2 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500,
                                 'save_config_overwrite': True},
                        model={'d_model': 128, 'growth_rate': 8, 'num_layers': 8, 'nhead': 8, 'num_decoder_layers': 3,
                               'dim_feedforward': 512, 'dropout': 0.3,
                               'dc': 8, 'cross_coverage': True, 'self_coverage': True,
                               'beam_size': 5, 'max_len': 200, 'alpha': 1.0, 'early_stopping':False, 'temperature': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': zipfile_path, 'test_year': 'test',
                              'train_batch_size': 8, 'eval_batch_size': 4,
                              'num_workers': 5, 'data_augmentation':augmentation, 'scale_aug':False}
                        )
    
    params_4 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500,
                                 'save_config_overwrite': True},
                        model={'d_model': 256, 'growth_rate': 8, 'num_layers': 16, 'nhead': 8, 'num_decoder_layers': 3,
                               'dim_feedforward': 256, 'dropout': 0.3,
                               'dc': 8, 'cross_coverage': True, 'self_coverage': True,
                               'beam_size': 5, 'max_len': 200, 'alpha': 1.0, 'early_stopping':False, 'temperature': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': zipfile_path, 'test_year': 'test',
                              'train_batch_size': 8, 'eval_batch_size': 4,
                              'num_workers': 5, 'data_augmentation':augmentation, 'scale_aug':False}
                        )
    
    params_5 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500,
                                 'save_config_overwrite': True},
                        model={'d_model': 64, 'growth_rate': 24, 'num_layers': 8, 'nhead': 8, 'num_decoder_layers': 1,
                               'dim_feedforward': 256, 'dropout': 0.3,
                               'dc': 32, 'cross_coverage': True, 'self_coverage': True,
                               'beam_size': 5, 'max_len': 200, 'alpha': 1.0, 'early_stopping':False, 'temperature': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': zipfile_path, 'test_year': 'test',
                              'train_batch_size': 8, 'eval_batch_size': 4,
                              'num_workers': 5, 'data_augmentation':augmentation, 'scale_aug':False}
                        )
    
    
    print(train_test_COMER_model(params_1_3))
    print(train_test_COMER_model(params_2_1))
    print(train_test_COMER_model(params_3_2))
    print(train_test_COMER_model(params_4))
    print(train_test_COMER_model(params_5))
    
