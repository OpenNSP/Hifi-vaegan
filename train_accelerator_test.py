
import multiprocessing
import os
import time
import modules.commons as commons
import utils
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch
from modules.models import MultiPeriodDiscriminator, TrainModel
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
accelerator = Accelerator()

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich import print

torch.backends.cudnn.benchmark = True
global_step = 0

progress = Progress(
    TextColumn("Running: "),
    BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn(),
    "•",
    TextColumn("[progress.description]{task.description}"),
    transient=True
    )

def train():
    global global_step
    hps = utils.get_hparams()
    torch.manual_seed(hps.train.seed)
    all_in_mem = hps.train.all_in_mem
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True, batch_size=hps.train.batch_size, collate_fn=TextAudioCollate())

    if accelerator.is_main_process:
        print("Hyperparameters:", hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps, all_in_mem=all_in_mem,vol_aug = False,is_slice=False)
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False, batch_size=1, pin_memory=False, drop_last=False, collate_fn=TextAudioCollate())

    net_g = TrainModel(hps.data.hop_length, hps.data.win_length, **hps.model)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g, skip_optimizer)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d, skip_optimizer)
        epoch_str = max(epoch_str, 1)
        name=utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
        global_step=int(name[name.rfind("_") + 1:name.rfind(".")]) + 1
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    train_loader, net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d = accelerator.prepare(train_loader, net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d)

    print("======= Start training =======")
    with progress:
        train_task = progress.add_task("Train", total=len(train_loader) - 1)
        for epoch in range(epoch_str, hps.train.epochs + 1):
            if epoch <= warmup_epoch:
                for param_group in optim_g.param_groups:
                    param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
                for param_group in optim_d.param_groups:
                    param_group['lr'] = hps.train.learning_rate / warmup_epoch * epoch
            
            net_g.train()
            net_d.train()
            for _, items in enumerate(train_loader):
                start_time = time.time()
                with accelerator.accumulate(net_g,net_d):
                    wav, lengths = items

                    mel = mel_spectrogram_torch(
                        wav,
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax)
                    
                    z, y_hat, (m, logs), commit_loss = net_g(wav)

                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.squeeze(1),
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.hop_length,
                        hps.data.win_length,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax
                    )
                        
                    wav = wav[:, None, :]
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wav, y_hat.detach())

                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_disc_all = loss_disc
                    
                    optim_d.zero_grad()
                    accelerator.backward(loss_disc_all)
                    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
                    optim_d.step()
                    
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wav, y_hat)
                    loss_mel = F.l1_loss(mel, y_hat_mel) * hps.train.c_mel
                    loss_wav = F.l1_loss(wav, y_hat) * hps.train.c_wav
                    loss_kl = kl_loss(logs, m) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    commit_loss = commit_loss * hps.train.c_vq
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_wav + commit_loss
                    
                    optim_g.zero_grad()
                    accelerator.backward(loss_gen_all)
                    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
                    optim_g.step()
                    

                if accelerator.is_main_process:
                    if global_step % hps.train.log_interval == 0:
                        lr = optim_g.param_groups[0]['lr']

                        scalar_dict = {"loss/g": loss_gen, "loss/d": loss_disc_all, "lr": lr, "grad_norm/g": grad_norm_g, "grad_norm/d": grad_norm_d}
                        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl, "loss/loss_wav":loss_wav, "loss/vq_loss": commit_loss})

                        image_dict = {
                            "slice/mel_org": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                        }

                        utils.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict)

                    if global_step % hps.train.eval_interval == 0:
                        evaluate(hps, net_g, eval_loader, writer)
                        utils.save_checkpoint(accelerator.unwrap_model(net_g) , optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                        utils.save_checkpoint(accelerator.unwrap_model(net_d), optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                        keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                        if keep_ckpts > 0:
                            utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                            print(f"Save checkpoint: G_{global_step}.pth D_{global_step}.pth | epoch={epoch}, step={global_step}, lr={optim_g.param_groups[0]['lr']:.5f}, loss_g={loss_gen.item():.2f}, loss_fm={loss_fm.item():.2f}, loss_mel={loss_mel.item():.2f}, loss_kl={loss_kl.item():.4f}, loss_wav={loss_wav.item():.2f}, vq_loss={commit_loss.item():.2f}")
                end_time = time.time()
                progress.update(train_task, advance=1, description=f"speed={1 / (end_time - start_time):.2f}it/s, epoch={epoch}, step={global_step}, lr={optim_g.param_groups[0]['lr']:.5f}, loss_g={loss_gen.item():.2f}, loss_fm={loss_fm.item():.2f}, loss_mel={loss_mel.item():.2f}, loss_kl={loss_kl.item():.2f}, loss_wav={loss_wav.item():.2f}, vq_loss={commit_loss.item():.4f}, grad_norm={grad_norm_g:.2f}")
                
                if accelerator.sync_gradients:
                    global_step += 1
                    
                accelerator.wait_for_everyone()
            scheduler_g.step()
            scheduler_d.step()
            progress.reset(train_task)          

def evaluate(hps, generator, eval_loader, writer):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        evaluate_task = progress.add_task("Evaluate", total=len(eval_loader) - 1)
        for batch_idx, items in enumerate(eval_loader):
            progress.update(evaluate_task, description=f"audio=_{batch_idx}")
            wav, length = items
            
            wav = wav.cuda(0)

            mel = mel_spectrogram_torch(
                wav,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax)

            z, y_hat, (m, logs), commit_loss = generator(wav)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            audio_dict.update({f"gen/audio_{batch_idx}": y_hat[0], f"gt/audio_{batch_idx}": wav[0]})
            progress.update(evaluate_task, advance=1)
        image_dict.update({"mel/gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()), "mel/gt": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
    progress.update(evaluate_task, description=f"Writing Summarize")
    utils.summarize(writer=writer, global_step=global_step, images=image_dict, audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate)
    progress.remove_task(evaluate_task)
    generator.train()

if __name__ == "__main__":
    train()
