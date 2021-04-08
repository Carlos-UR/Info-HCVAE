import torch
import torch.nn as nn
import pdb
from models import DiscreteVAE, return_mask_lengths


class VAETrainer(object):
    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device

        self.vae = DiscreteVAE(args).to(self.device)
        params = filter(lambda p: p.requires_grad, self.vae.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr)

        self.loss_q_rec = 0
        self.loss_a_rec = 0
        self.loss_zq_kl = 0
        self.loss_za_kl = 0
        self.loss_info = 0

    def train(self, c_ids, q_ids, a_ids, start_positions, end_positions):
        self.vae = self.vae.train()

        # Forward
        loss, \
        loss_q_rec, loss_a_rec, \
        loss_zq_kl, loss_za_kl, \
        loss_info \
        = self.vae(c_ids, q_ids, a_ids, start_positions, end_positions)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Step
        self.optimizer.step()

        self.loss_q_rec = loss_q_rec.item()
        self.loss_a_rec = loss_a_rec.item()
        self.loss_zq_kl = loss_zq_kl.item()
        self.loss_za_kl = loss_za_kl.item()
        self.loss_info = loss_info.item()

    def generate_posterior(self, c_ids, q_ids, a_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            _, _, zq, _, za = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            q_ids, start_positions, end_positions = self.vae.generate(zq, za, c_ids)
        return q_ids, start_positions, end_positions, zq

    def generate_answer_logits(self, c_ids, q_ids, a_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            _, _, zq, _, za = self.vae.posterior_encoder(c_ids, q_ids, a_ids)
            start_logits, end_logits = self.vae.return_answer_logits(zq, za, c_ids)
        return start_logits, end_logits

    def generate_prior(self, c_ids):
        self.vae = self.vae.eval()
        with torch.no_grad():
            _, _, zq, _, za = self.vae.prior_encoder(c_ids)
            q_ids, start_positions, end_positions = self.vae.generate(zq, za, c_ids)
        return q_ids, start_positions, end_positions, zq    

    # [Update] Ahora guarda el estado actual del objeto (todas las variables del objeto).
    def save(self, filename, epoch, f1, bleu, em):
        params = {
            'state_dict': self.vae.state_dict(),
            'args': self.args,
            'optimizer': self.optimizer.state_dict(),
            'loss_q_rec': self.loss_q_rec,
            'loss_a_rec': self.loss_a_rec,
            'loss_zq_kl': self.loss_zq_kl,
            'loss_za_kl': self.loss_za_kl,
            'loss_info': self.loss_info,
            'epoch': epoch,
            'f1': f1,
            'bleu': bleu,
            'em': em
        }
        torch.save(params, filename)
    

    # [New] Función que carga un modelo así (Y todas las variables que forman el estado en el que se guardó).
    #     - Carga la última época almacenada con save_epoch y la devuelve.
    def loadd(self, foldername):
        checkpoint = torch.load(f"{foldername}/checkpoint.pt")

        self.vae.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.loss_q_rec = checkpoint['loss_q_rec']
        self.loss_a_rec = checkpoint['loss_a_rec']
        self.loss_zq_kl = checkpoint['loss_zq_kl']
        self.loss_za_kl = checkpoint['loss_za_kl']
        self.loss_info = checkpoint['loss_info']

        print(f"Loading model trained in {checkpoint['epoch']} epochs.")

        return checkpoint['epoch']
    @staticmethod
    def get_best_f1(foldername):
        return torch.load(f"{foldername}/best_f1_model.pt")['f1']

    @staticmethod
    def get_best_bleu(foldername):
        return torch.load(f"{foldername}/best_bleu_model.pt")['bleu']

    @staticmethod
    def get_best_em(foldername):
        return torch.load(f"{foldername}/best_f1_model.pt")['em']

    @staticmethod
    def load_measures(fn):
        return VAETrainer.get_best_f1(fn), VAETrainer.get_best_bleu(fn), VAETrainer.get_best_em(fn)