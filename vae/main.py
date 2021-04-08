import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import BertTokenizer

from eval import eval_vae
from trainer import VAETrainer
from utils import batch_to_device, get_harv_data_loader, get_squad_data_loader

import uuid
import mlflow
#import mlflow_utils 
from mlflow_utils import MLFLowLogger

def init_mlflow(args, mlflow_tracking_uri):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    id = uuid.uuid4().hex
    mlflow_experiment_id = mlflow.create_experiment('Ray_Test_{}'.format(id))
    print(f"Creating MLFLow experiment with id: {mlflow_experiment_id}.")

    config = {
      "epochs" : args.epochs,
      "batch_size" : args.batch_size,
      "lr" : args.lr,
      "mlflow_experiment_id": mlflow_experiment_id,
      "mlflow_tracking_uri": mlflow_tracking_uri
    }

    return MLFLowLogger(config)

# Calcula los resultados del modelo, los imprime por pantalla y los devuelve.

def eval_measures(epoch, args, trainer, eval_data):
    metric_dict, bleu, _ = eval_vae(epoch, args, trainer, eval_data)
    _str = '{}-th Epochs BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
    _str = _str.format(epoch, bleu, metric_dict["exact_match"], metric_dict["f1"])
    return metric_dict["f1"], metric_dict["exact_match"], bleu*100, _str

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    train_loader, _, _ = get_squad_data_loader(tokenizer, args.train_dir,
                                         shuffle=True, args=args)
    eval_data = get_squad_data_loader(tokenizer, args.dev_dir,
                                      shuffle=False, args=args)

    args.device = torch.cuda.current_device()

    trainer = VAETrainer(args)
	
    loss_log1 = tqdm(total=0, bar_format='{desc}', position=2)
    loss_log2 = tqdm(total=0, bar_format='{desc}', position=3)
    eval_log = tqdm(total=0, bar_format='{desc}', position=5)
    best_eval_log = tqdm(total=0, bar_format='{desc}', position=6)

	  # Cargar checkpoint
    if args.load_checkpoint:
      epochs = trainer.loadd(args.model_dir)
      best_f1, best_bleu, best_em = VAETrainer.load_measures(args.model_dir)
      print(f"The current best measures are: F1  = {best_f1}, BLEU = {best_bleu} and EM = {best_em}.")
    else:
      epochs = -1
      best_bleu, best_em, best_f1 = 0.0, 0.0, 0.0

    print("MODEL DIR: " + args.model_dir)
    mlflow_logger = init_mlflow(args, f"{args.model_dir}/mllogs")
    for epoch in trange(int(args.epochs), desc="Epoch", position=0):
      if epoch <= epochs:
        print(f"jumping epoch {epoch}...")
      else:
        for batch in tqdm(train_loader, desc="Train iter", leave=False, position=1):
          c_ids, q_ids, a_ids, start_positions, end_positions \
          = batch_to_device(batch, args.device)
          trainer.train(c_ids, q_ids, a_ids, start_positions, end_positions)
          
          str1 = 'Q REC : {:06.4f} A REC : {:06.4f}'
          str2 = 'ZQ KL : {:06.4f} ZA KL : {:06.4f} INFO : {:06.4f}'
          str1 = str1.format(float(trainer.loss_q_rec), float(trainer.loss_a_rec))
          str2 = str2.format(float(trainer.loss_zq_kl), float(trainer.loss_za_kl), float(trainer.loss_info))
          loss_log1.set_description_str(str1)
          loss_log2.set_description_str(str2)

        if epoch >= 0:
          f1, em, bleu, _str = eval_measures(epoch, args, trainer, eval_data)
          eval_log.set_description_str(_str)
          result = {
            "epoch" : epoch,
            "em" : em,
            "f1" : f1,
            "bleu" : bleu
          }
          mlflow_logger.on_result(result)
          if em > best_em:
            best_em = em
          if f1 > best_f1:
            best_f1 = f1
            trainer.save(os.path.join(args.model_dir, "best_f1_model.pt"), epoch, f1, bleu, em)
          if bleu > best_bleu:
            best_bleu = bleu
            trainer.save(os.path.join(args.model_dir, "best_bleu_model.pt"), epoch, f1, bleu, em)
          trainer.save(os.path.join(args.model_dir, "checkpoint.pt"), epoch, f1, bleu, em)

          _str = 'BEST BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
          _str = _str.format(best_bleu, best_em, best_f1)
          best_eval_log.set_description_str(_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')

    # [New] Nuevo argumento para señalar que queremos cargar un checkpoint.
    parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_true')
    parser.add_argument('--train_dir', default='../data/squad/train-v1.1.json')
    parser.add_argument('--dev_dir', default='../data/squad/my_dev.json')
    
    parser.add_argument("--max_c_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=64, type=int, help="max query length")

    parser.add_argument("--model_dir", default="../save/vae-checkpoint", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="lr")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--clip", default=5.0, type=float, help="max grad norm")

    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument('--enc_nhidden', type=int, default=300)
    parser.add_argument('--enc_nlayers', type=int, default=1)
    parser.add_argument('--enc_dropout', type=float, default=0.2)
    parser.add_argument('--dec_a_nhidden', type=int, default=300)
    parser.add_argument('--dec_a_nlayers', type=int, default=1)
    parser.add_argument('--dec_a_dropout', type=float, default=0.2)
    parser.add_argument('--dec_q_nhidden', type=int, default=900)
    parser.add_argument('--dec_q_nlayers', type=int, default=2)
    parser.add_argument('--dec_q_dropout', type=float, default=0.3)
    parser.add_argument('--nzqdim', type=int, default=50)
    parser.add_argument('--nza', type=int, default=20)
    parser.add_argument('--nzadim', type=int, default=10)
    parser.add_argument('--lambda_kl', type=float, default=0.1)
    parser.add_argument('--lambda_info', type=float, default=1.0)

    args = parser.parse_args()

    if args.debug:
        args.model_dir = "./dummy"
    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)