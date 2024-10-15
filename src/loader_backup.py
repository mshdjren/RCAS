# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/loader.py

from os.path import dirname, abspath, exists, join
import sys
import glob
import json
import os
import random
import warnings

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import wandb

from data_util import Dataset_
from utils.style_ops import grid_sample_gradfix
from utils.style_ops import conv2d_gradfix
from metrics.inception_net import InceptionV3
from sync_batchnorm.batchnorm import convert_model
from worker import WORKER
import utils.log as log
import utils.losses as losses
import utils.ckpt as ckpt
import utils.misc as misc
import utils.custom_ops as custom_ops
import models.model as model
import metrics.preparation as pp


def load_worker(local_rank, cfgs, gpus_per_node, run_name, hdf5_path):
    # -----------------------------------------------------------------------------
    # define default variables for loading ckpt or evaluating the trained GAN model.
    # -----------------------------------------------------------------------------
    load_train_dataset = cfgs.RUN.train + cfgs.RUN.GAN_train + cfgs.RUN.GAN_test
    len_eval_metrics = 0 if cfgs.RUN.eval_metrics == ["none"] else len(cfgs.RUN.eval_metrics)
    load_eval_dataset = len_eval_metrics + cfgs.RUN.save_real_images + cfgs.RUN.k_nearest_neighbor + \
        cfgs.RUN.frequency_analysis + cfgs.RUN.tsne_analysis + cfgs.RUN.intra_class_fid
    train_sampler, eval_sampler = None, None
    step, epoch, topk, best_step, best_fid, best_ckpt_path, lecam_emas, is_best = \
        0, 0, cfgs.OPTIMIZATION.batch_size, 0, None, None, None, False
    mu, sigma, real_feats, eval_model, num_rows, num_cols = None, None, None, None, 10, 8
    aa_p = cfgs.AUG.ada_initial_augment_p
    if cfgs.AUG.ada_initial_augment_p != "N/A":
        aa_p = cfgs.AUG.ada_initial_augment_p
    else:
        aa_p = cfgs.AUG.apa_initial_augment_p

    loss_list_dict = {"gen_loss": [], "dis_loss": [], "cls_loss": []}
    num_eval = {}
    metric_dict_during_train = {}
    if "none" in cfgs.RUN.eval_metrics:
        cfgs.RUN.eval_metrics = []
    if "is" in cfgs.RUN.eval_metrics:
        metric_dict_during_train.update({"IS": [], "Top1_acc": [], "Top5_acc": []})
    if "fid" in cfgs.RUN.eval_metrics:
        metric_dict_during_train.update({"FID": []})
    if "prdc" in cfgs.RUN.eval_metrics:
        metric_dict_during_train.update({"Improved_Precision": [], "Improved_Recall": [], "Density":[], "Coverage": []})

    # -----------------------------------------------------------------------------
    # determine cuda, cudnn, and backends settings.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.fix_seed:
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        cudnn.benchmark, cudnn.deterministic = True, False

    if cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]:
        # Improves training speed
        conv2d_gradfix.enabled = True
        # Avoids errors with the augmentation pipe
        grid_sample_gradfix.enabled = True
        if cfgs.RUN.mixed_precision:
            # Allow PyTorch to internally use tf32 for matmul
            torch.backends.cuda.matmul.allow_tf32 = False
            # Allow PyTorch to internally use tf32 for convolutions
            torch.backends.cudnn.allow_tf32 = False

    # -----------------------------------------------------------------------------
    # initialize all processes and fix seed of each process
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        global_rank = cfgs.RUN.current_node * (gpus_per_node) + local_rank
        print("Use GPU: {global_rank} for training.".format(global_rank=global_rank))
        misc.setup(global_rank, cfgs.OPTIMIZATION.world_size, cfgs.RUN.backend)
        torch.cuda.set_device(local_rank)
    else:
        global_rank = local_rank

    misc.fix_seed(cfgs.RUN.seed + global_rank)

    # -----------------------------------------------------------------------------
    # Intialize python logger.
    # -----------------------------------------------------------------------------
    if local_rank == 0:
        logger = log.make_logger(cfgs.RUN.save_dir, run_name, None)
        if cfgs.RUN.ckpt_dir is not None and cfgs.RUN.freezeD == -1:
            folder_hier = cfgs.RUN.ckpt_dir.split("/")
            if folder_hier[-1] == "":
                folder_hier.pop()
            logger.info("Run name : {run_name}".format(run_name=folder_hier.pop()))
        else:
            logger.info("Run name : {run_name}".format(run_name=run_name))
        for k, v in cfgs.super_cfgs.items():
            logger.info("cfgs." + k + " =")
            logger.info(json.dumps(vars(v), indent=2))
    else:
        logger = None

    # -----------------------------------------------------------------------------
    # load train and evaluation datasets.
    # -----------------------------------------------------------------------------
    if load_train_dataset:
        if local_rank == 0:
            logger.info("Load {name} train dataset for training.".format(name=cfgs.DATA.name))
        train_dataset = Dataset_(data_name=cfgs.DATA.name,
                                 data_dir=cfgs.RUN.data_dir,
                                 train=True,
                                 crop_long_edge=cfgs.PRE.crop_long_edge,
                                 resize_size=cfgs.PRE.resize_size,
                                 resizer=None if hdf5_path is not None else cfgs.RUN.pre_resizer,
                                 random_flip=cfgs.PRE.apply_rflip,
                                 normalize=True,
                                 hdf5_path=hdf5_path,
                                 load_data_in_memory=cfgs.RUN.load_data_in_memory)
        if local_rank == 0:
            logger.info("Train dataset size: {dataset_size}".format(dataset_size=len(train_dataset)))
    else:
        train_dataset = None

    if  load_eval_dataset:
        if local_rank == 0:
            logger.info("Load {name} {ref} dataset for evaluation.".format(name=cfgs.DATA.name, ref=cfgs.RUN.ref_dataset))
        eval_dataset = Dataset_(data_name=cfgs.DATA.name,
                                data_dir=cfgs.RUN.data_dir,
                                train=True if cfgs.RUN.ref_dataset == "train" else False,
                                crop_long_edge=False if cfgs.DATA.name in cfgs.MISC.no_proc_data else True,
                                resize_size=None if cfgs.DATA.name in cfgs.MISC.no_proc_data else cfgs.DATA.img_size,
                                resizer=cfgs.RUN.pre_resizer,
                                random_flip=False,
                                hdf5_path=None,
                                normalize=True,
                                load_data_in_memory=False)
        if local_rank == 0:
            logger.info("Eval dataset size: {dataset_size}".format(dataset_size=len(eval_dataset)))
    else:
        eval_dataset = None

    # -----------------------------------------------------------------------------
    # define a distributed sampler for DDP train and evaluation.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.distributed_data_parallel:
        cfgs.OPTIMIZATION.batch_size = cfgs.OPTIMIZATION.batch_size//cfgs.OPTIMIZATION.world_size
        if cfgs.RUN.train:
            train_sampler = DistributedSampler(train_dataset,
                                            num_replicas=cfgs.OPTIMIZATION.world_size,
                                            rank=local_rank,
                                            shuffle=True,
                                            drop_last=True)
            topk = cfgs.OPTIMIZATION.batch_size

        if load_eval_dataset:
            eval_sampler = DistributedSampler(eval_dataset,
                                            num_replicas=cfgs.OPTIMIZATION.world_size,
                                            rank=local_rank,
                                            shuffle=False,
                                            drop_last=False)

    cfgs.OPTIMIZATION.basket_size = cfgs.OPTIMIZATION.batch_size*\
                                    cfgs.OPTIMIZATION.acml_steps*\
                                    cfgs.OPTIMIZATION.d_updates_per_step
    #change
    # -----------------------------------------------------------------------------
    # define dataloaders for train and evaluation.
    # -----------------------------------------------------------------------------
    if cfgs.RUN.mas or cfgs.RUN.retrain or cfgs.RUN.split_loader:
        if local_rank == 0:
            logger.info("Load {name} train dataset for fine-training or retraining.".format(name=cfgs.DATA.name))
        from mas_utils.custom_dataset import get_unlearn_loader
        _, train_dataloader, _, eval_dataloader = get_unlearn_loader(trainset= train_dataset, testset= eval_dataset, \
                                                                     target_classes=cfgs.RUN.target_classes, remain_classes = cfgs.RUN.remain_classes, \
                                                                     cfgs=cfgs, logger= logger,\
                                                                     load_train_dataset = load_train_dataset, load_eval_dataset = load_eval_dataset)
        # if cfgs.RUN.tsne_analysis:
        #     if load_eval_dataset:
        #         eval_dataloader = DataLoader(dataset=eval_dataset,
        #                                     batch_size=cfgs.OPTIMIZATION.batch_size,
        #                                     shuffle=False,
        #                                     pin_memory=True,
        #                                     num_workers=cfgs.RUN.num_workers,
        #                                     sampler=eval_sampler,
        #                                     drop_last=False)
    else:
        if load_train_dataset:
            train_dataloader = DataLoader(dataset=train_dataset,
                                        batch_size=cfgs.OPTIMIZATION.basket_size,
                                        shuffle=(train_sampler is None),
                                        pin_memory=True,
                                        num_workers=cfgs.RUN.num_workers,
                                        sampler=train_sampler,
                                        drop_last=True,
                                        persistent_workers=True)
        else:
            train_dataloader = None

        if load_eval_dataset:
            eval_dataloader = DataLoader(dataset=eval_dataset,
                                        batch_size=cfgs.OPTIMIZATION.batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=cfgs.RUN.num_workers,
                                        sampler=eval_sampler,
                                        drop_last=False)
        else:
            eval_dataloader = None


    # -----------------------------------------------------------------------------
    # load a generator and a discriminator
    # if cfgs.MODEL.apply_g_ema is True, load an exponential moving average generator (Gen_ema).
    # -----------------------------------------------------------------------------
    #change
    Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis, ema =\
        model.load_generator_discriminator(DATA=cfgs.DATA,
                                           OPTIMIZATION=cfgs.OPTIMIZATION,
                                           MODEL=cfgs.MODEL,
                                           STYLEGAN=cfgs.STYLEGAN,
                                           MODULES=cfgs.MODULES,
                                           RUN=cfgs.RUN,
                                           DYNAMIC=cfgs.DYNAMIC,
                                           device=local_rank,
                                           logger=logger)

    if local_rank != 0:
        custom_ops.verbosity = "none"

    # -----------------------------------------------------------------------------
    # define optimizers for adversarial training
    # -----------------------------------------------------------------------------
    cfgs.define_optimizer(Gen, Dis)

    # -----------------------------------------------------------------------------
    # load the generator and the discriminator from a checkpoint if possible
    # -----------------------------------------------------------------------------
    if cfgs.RUN.ckpt_dir is not None:
        if local_rank == 0:
            os.remove(join(cfgs.RUN.save_dir, "logs", run_name + ".log"))
        run_name, step, epoch, topk, aa_p, best_step, best_fid, best_ckpt_path, lecam_emas, logger =\
            ckpt.load_StudioGAN_ckpts(ckpt_dir=cfgs.RUN.ckpt_dir,
                                      load_best=cfgs.RUN.load_best,
                                      Gen=Gen,
                                      Dis=Dis,
                                      g_optimizer=cfgs.OPTIMIZATION.g_optimizer,
                                      d_optimizer=cfgs.OPTIMIZATION.d_optimizer,
                                      run_name=run_name,
                                      apply_g_ema=cfgs.MODEL.apply_g_ema,
                                      Gen_ema=Gen_ema,
                                      ema=ema,
                                      is_train=cfgs.RUN.train,
                                      RUN=cfgs.RUN,
                                      logger=logger,
                                      global_rank=global_rank,
                                      device=local_rank,
                                      cfg_file=cfgs.RUN.cfg_file)

        if topk == "initialize":
            topk == cfgs.OPTIMIZATION.batch_size
        if cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]:
            ema.ema_rampup = "N/A" # disable EMA rampup
            if cfgs.MODEL.backbone == "stylegan3" and cfgs.STYLEGAN.stylegan3_cfg == "stylegan3-r":
                cfgs.STYLEGAN.blur_init_sigma = "N/A" # disable blur rampup
        if cfgs.AUG.apply_ada:
            cfgs.AUG.ada_kimg = 100 # make ADA react faster at the beginning

    if cfgs.RUN.ckpt_dir is None or cfgs.RUN.freezeD != -1:
        if local_rank == 0:
            cfgs.RUN.ckpt_dir = ckpt.make_ckpt_dir(join(cfgs.RUN.save_dir, "checkpoints", run_name))
        dict_dir = join(cfgs.RUN.save_dir, "statistics", run_name)
        loss_list_dict = misc.load_log_dicts(directory=dict_dir, file_name="losses.npy", ph=loss_list_dict)
        metric_dict_during_train = misc.load_log_dicts(directory=dict_dir, file_name="metrics.npy", ph=metric_dict_during_train)

    # -----------------------------------------------------------------------------
    # prepare parallel training
    # -----------------------------------------------------------------------------
    if cfgs.OPTIMIZATION.world_size > 1:
        Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis =\
        model.prepare_parallel_training(Gen=Gen,
                                        Gen_mapping=Gen_mapping,
                                        Gen_synthesis=Gen_synthesis,
                                        Dis=Dis,
                                        Gen_ema=Gen_ema,
                                        Gen_ema_mapping=Gen_ema_mapping,
                                        Gen_ema_synthesis=Gen_ema_synthesis,
                                        MODEL=cfgs.MODEL,
                                        world_size=cfgs.OPTIMIZATION.world_size,
                                        distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                        synchronized_bn=cfgs.RUN.synchronized_bn,
                                        apply_g_ema=cfgs.MODEL.apply_g_ema,
                                        device=local_rank)

    # -----------------------------------------------------------------------------
    # load a pre-trained network (InceptionV3, SwAV, DINO, or Swin-T)
    # -----------------------------------------------------------------------------
    if cfgs.DATA.name in ["ImageNet", "Baby_ImageNet", "Papa_ImageNet", "Grandpa_ImageNet"]:
        num_eval = {"train": 50000, "valid": len(eval_dataloader.dataset)}
    else:
        if eval_dataloader is not None:
            num_eval[cfgs.RUN.ref_dataset] = len(eval_dataloader.dataset)
        else:
            num_eval["train"], num_eval["valid"], num_eval["test"] = 50000, 50000, 50000

    if len(cfgs.RUN.eval_metrics) or cfgs.RUN.intra_class_fid:
        eval_model = pp.LoadEvalModel(eval_backbone=cfgs.RUN.eval_backbone,
                                      post_resizer=cfgs.RUN.post_resizer,
                                      world_size=cfgs.OPTIMIZATION.world_size,
                                      distributed_data_parallel=cfgs.RUN.distributed_data_parallel,
                                      device=local_rank)

    if "fid" in cfgs.RUN.eval_metrics:
        mu, sigma = pp.prepare_moments(data_loader=eval_dataloader,
                                       eval_model=eval_model,
                                       quantize=True,
                                       cfgs=cfgs,
                                       logger=logger,
                                       device=local_rank)

    if "prdc" in cfgs.RUN.eval_metrics:
        if cfgs.RUN.distributed_data_parallel:
            prdc_sampler = DistributedSampler(eval_dataset,
                                              num_replicas=cfgs.OPTIMIZATION.world_size,
                                              rank=local_rank,
                                              shuffle=True,
                                              drop_last=False)
        else:
            prdc_sampler = None

        prdc_dataloader = DataLoader(dataset=eval_dataset,
                                     batch_size=cfgs.OPTIMIZATION.batch_size,
                                     shuffle=(prdc_sampler is None),
                                     pin_memory=True,
                                     num_workers=cfgs.RUN.num_workers,
                                     sampler=prdc_sampler,
                                     drop_last=False)

        real_feats = pp.prepare_real_feats(data_loader=prdc_dataloader,
                                           eval_model=eval_model,
                                           num_feats=num_eval[cfgs.RUN.ref_dataset],
                                           quantize=True,
                                           cfgs=cfgs,
                                           logger=logger,
                                           device=local_rank)

    if cfgs.RUN.calc_is_ref_dataset:
        pp.calculate_ins(data_loader=eval_dataloader,
                         eval_model=eval_model,
                         quantize=True,
                         splits=1,
                         cfgs=cfgs,
                         logger=logger,
                         device=local_rank)
        
    #change 
    if cfgs.RUN.mas:

        if local_rank == 0:
            logger.info("Prepare measuring weight importances with Memory Aware Synapses")

        from mas_utils.mas import Mas
        from mas_utils.mas import init_reg_params

        Gen_mas = Mas(source=Gen)
        Dis_mas = Mas(source=Dis)

        print("===================initializing omega values for Generator===================")
        Gen_mas, Gen_block = init_reg_params(Gen_mas, local_rank, cfgs.RUN.selectG_blocks, cfgs.RUN.selectG_layers)
        print("===================initializing omega values for Discriminator===================")
        Dis_mas, Dis_block = init_reg_params(Dis_mas, local_rank, cfgs.RUN.selectD_blocks, cfgs.RUN.selectD_layers, cfgs.RUN.linear1, cfgs.RUN.linear2, cfgs.RUN.last_conv)

        #calculate omega values for each gen & disc

        from mas_utils.mas_init_intersection import init_no_intersection

        Gen_mas, Dis_mas = init_no_intersection(Gen_mas = Gen_mas, Gen_block = Gen_block, Dis_mas = Dis_mas, Dis_block = Dis_block, cfgs = cfgs, global_rank = global_rank, local_rank = local_rank,\
                                            logger = logger, change = True, dis_change = False)
        
        # Gen_mas, Dis_mas = init_no_intersection_channelwise(Gen_mas = Gen_mas, Gen_block = Gen_block, Dis_mas = Dis_mas, Dis_block = Dis_block, cfgs = cfgs, global_rank = global_rank, local_rank = local_rank,\
        #                                             logger = logger, change = True, dis_change = True)
        betas_g = betas_d = [cfgs.OPTIMIZATION.beta1, cfgs.OPTIMIZATION.beta2]
        eps_ = 1e-6

        cfgs.OPTIMIZATION.g_optimizer = torch.optim.Adam(params=Gen_mas.source.parameters(),
                                                        lr=cfgs.OPTIMIZATION.g_lr,
                                                        betas=betas_g,
                                                        weight_decay=cfgs.OPTIMIZATION.g_weight_decay,
                                                        eps=eps_) 

        cfgs.OPTIMIZATION.d_optimizer = torch.optim.Adam(params=Dis_mas.source.parameters(),
                                                        lr=cfgs.OPTIMIZATION.d_lr,
                                                        betas=betas_d,
                                                        weight_decay=cfgs.OPTIMIZATION.d_weight_decay,
                                                        eps=eps_)

        g_scheduler = torch.optim.lr_scheduler.StepLR(cfgs.OPTIMIZATION.g_optimizer, step_size=10000, gamma=0.1)
        d_scheduler = torch.optim.lr_scheduler.StepLR(cfgs.OPTIMIZATION.d_optimizer, step_size=10000, gamma=0.1)

    else:
        gen_importance = None
        dis_importance = None

    # -----------------------------------------------------------------------------
    # initialize WORKER for training and evaluating GAN
    # -----------------------------------------------------------------------------
    #change 
    worker = WORKER(
        cfgs=cfgs,
        run_name=run_name,
        Gen=Gen,
        Gen_mapping=Gen_mapping,
        Gen_synthesis=Gen_synthesis,
        Dis=Dis,
        Gen_ema=Gen_ema,
        Gen_ema_mapping=Gen_ema_mapping,
        Gen_ema_synthesis=Gen_ema_synthesis,
        ema=ema,
        eval_model=eval_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        global_rank=global_rank,
        local_rank=local_rank,
        mu=mu,
        sigma=sigma,
        real_feats=real_feats,
        logger=logger,
        aa_p=aa_p,
        best_step=best_step,
        best_fid=best_fid,
        best_ckpt_path=best_ckpt_path,
        lecam_emas=lecam_emas,
        num_eval=num_eval,
        loss_list_dict=loss_list_dict,
        metric_dict_during_train=metric_dict_during_train,
        gen_importance = gen_importance,
        dis_importance = dis_importance
    )

    # -----------------------------------------------------------------------------
    # train GAN until "total_steps" generator updates
    # -----------------------------------------------------------------------------
    if cfgs.RUN.train:
        if global_rank == 0:
            logger.info("Start training!")

        worker.training, worker.topk = True, topk
        worker.prepare_train_iter(epoch_counter=epoch)
        while step <= cfgs.OPTIMIZATION.total_steps:
            #change        
            from models.dynamic_deep_conv import Conv2dLayer
            from models.dynamic_deep_conv import DiscriminatorEpilogue
            #Adjusting capacity of discriminator on-the-fly
            if cfgs.DYNAMIC.occupy_start > 0 and cfgs.DYNAMIC.occupy_end > 0:
                current_occupy_ratio = ((cfgs.DYNAMIC.occupy_end - cfgs.DYNAMIC.occupy_start) / cfgs.OPTIMIZATION.total_steps) * (step / 1e3) + cfgs.DYNAMIC.occupy_start
                for m in Dis.modules():
                    if hasattr(m, 'occupy_ratio') and isinstance(m, Conv2dLayer):
                        m.occupy_ratio = current_occupy_ratio
                    if isinstance(m, DiscriminatorEpilogue):
                        m.conv.occupy_ratio = current_occupy_ratio
                        # m.fc.occupy_ratio = current_occupy_ratio
                        # m.out.occupy_ratio = current_occupy_ratio

            #change
            if cfgs.RUN.loss_check == False:
                if cfgs.OPTIMIZATION.d_first:
                    real_cond_loss, dis_acml_loss,_,_,_ = worker.train_discriminator(current_step=step)
                    gen_acml_loss = worker.train_generator(current_step=step)
                else:
                    gen_acml_loss = worker.train_generator(current_step=step)
                    real_cond_loss, dis_acml_loss,_,_,_ = worker.train_discriminator(current_step=step)

                if global_rank == 0 and (step + 1) % cfgs.RUN.print_freq == 0:
                    worker.log_train_statistics(current_step=step,
                                                real_cond_loss=real_cond_loss,
                                                gen_acml_loss=gen_acml_loss,
                                                dis_acml_loss=dis_acml_loss)
            else:
                if cfgs.OPTIMIZATION.d_first:
                    real_cond_loss, dis_acml_loss, dis_acml_loss_real, dis_acml_loss_fake, gp_check = worker.train_discriminator(current_step=step)
                    gen_acml_loss = worker.train_generator(current_step=step)
                else:
                    gen_acml_loss = worker.train_generator(current_step=step)
                    real_cond_loss, dis_acml_loss, dis_acml_loss_real, dis_acml_loss_fake, gp_check = worker.train_discriminator(current_step=step)

                if global_rank == 0 and (step + 1) % cfgs.RUN.print_freq == 0:
                    worker.log_train_statistics(current_step=step,
                                                real_cond_loss=real_cond_loss,
                                                gen_acml_loss=gen_acml_loss,
                                                dis_acml_loss=dis_acml_loss,
                                                dis_acml_loss_real=dis_acml_loss_real,
                                                dis_acml_loss_fake=dis_acml_loss_fake,
                                                gp_check=gp_check                                            
                                                )

            step += 1

            if cfgs.LOSS.apply_topk:
                if (epoch + 1) == worker.epoch_counter:
                    epoch += 1
                    worker.topk = losses.adjust_k(current_k=worker.topk,
                                                  topk_gamma=cfgs.LOSS.topk_gamma,
                                                  inf_k=int(cfgs.OPTIMIZATION.batch_size * cfgs.LOSS.topk_nu))

            if step % cfgs.RUN.save_freq == 0:
                # visuailize fake images
                if global_rank == 0:
                   worker.visualize_fake_images(num_cols=num_cols, current_step=step)

                # evaluate GAN for monitoring purpose
                if len(cfgs.RUN.eval_metrics) :
                    is_best = worker.evaluate(step=step, metrics=cfgs.RUN.eval_metrics, writing=True, training=True)

                # save GAN in "./checkpoints/RUN_NAME/*"
                if global_rank == 0:
                    worker.save(step=step, is_best=is_best)

                # stop processes until all processes arrive
                if cfgs.RUN.distributed_data_parallel:
                    dist.barrier(worker.group)

        if global_rank == 0:
            logger.info("End of training!")

    # -----------------------------------------------------------------------------
    # re-evaluate the best GAN and conduct ordered analyses
    # -----------------------------------------------------------------------------
    worker.training, worker.epoch_counter = False, epoch
    worker.gen_ctlr.standing_statistics = cfgs.RUN.standing_statistics
    worker.gen_ctlr.standing_max_batch = cfgs.RUN.standing_max_batch
    worker.gen_ctlr.standing_step = cfgs.RUN.standing_step

    if global_rank == 0:
        best_step = ckpt.load_best_model(ckpt_dir=cfgs.RUN.ckpt_dir,
                                         Gen=Gen,
                                         Dis=Dis,
                                         apply_g_ema=cfgs.MODEL.apply_g_ema,
                                         Gen_ema=Gen_ema,
                                         ema=ema)
    if len(cfgs.RUN.eval_metrics):
        for e in range(cfgs.RUN.num_eval):
            if global_rank == 0:
                print(""), logger.info("-" * 80)
            _ = worker.evaluate(step=best_step, metrics=cfgs.RUN.eval_metrics, writing=False, training=False)

    if cfgs.RUN.save_real_images:
        if global_rank == 0: print(""), logger.info("-" * 80)
        worker.save_real_images()

    if cfgs.RUN.save_fake_images:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.save_fake_images(num_images=cfgs.RUN.save_fake_images_num)

    if cfgs.RUN.vis_fake_images:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.visualize_fake_images(num_cols=num_cols, current_step=best_step)

    if cfgs.RUN.k_nearest_neighbor:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.run_k_nearest_neighbor(dataset=eval_dataset, num_rows=num_rows, num_cols=num_cols)

    if cfgs.RUN.interpolation:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.run_linear_interpolation(num_rows=num_rows, num_cols=num_cols, fix_z=True, fix_y=False)
        worker.run_linear_interpolation(num_rows=num_rows, num_cols=num_cols, fix_z=False, fix_y=True)

    if cfgs.RUN.frequency_analysis:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.run_frequency_analysis(dataloader=eval_dataloader)

    if cfgs.RUN.tsne_analysis:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.run_tsne(dataloader=eval_dataloader)

    if cfgs.RUN.intra_class_fid:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.calculate_intra_class_fid(dataset=eval_dataset)

    if cfgs.RUN.semantic_factorization:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.run_semantic_factorization(num_rows=cfgs.RUN.num_semantic_axis,
                                          num_cols=num_cols,
                                          maximum_variations=cfgs.RUN.maximum_variations)
    if cfgs.RUN.GAN_train:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.compute_GAN_train_or_test_classifier_accuracy_score(GAN_train=True, GAN_test=False)

    if cfgs.RUN.GAN_test:
        if global_rank == 0:
            print(""), logger.info("-" * 80)
        worker.compute_GAN_train_or_test_classifier_accuracy_score(GAN_train=False, GAN_test=True)

    if global_rank == 0:
        wandb.finish()
