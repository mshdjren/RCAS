import torch
import torch.nn as nn
from os.path import join
import numpy as np

import utils.misc as misc
import utils.sample as sample


def get_activation(name, reg_params):
    def hook(model, input, output):
        reg_params[name] = output.detach()
    return hook


# def init_no_intersection(Gen, Gen_block, Dis_mas, Dis_block, cfgs, global_rank, local_rank, logger, change = False, dis_change = False):
def init_no_intersection(Gen, Gen_block, Gen_reg_params, Gen_reg_params_intersect, \
                         Dis, Dis_block, Dis_reg_params, Dis_reg_params_intersect, \
                         cfgs, global_rank, local_rank, logger, run_name, change = False, dis_change = False):

    is_stylegan = cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]
    adc_fake = False


    misc.make_GAN_untrainable(Gen, None, Dis)

    fake_images, fake_labels, fake_images_eps, trsp_cost, ws, _, _ = sample.generate_images(
        z_prior=cfgs.MODEL.z_prior,
        truncation_factor=-1.0,
        batch_size=cfgs.OPTIMIZATION.batch_size,
        z_dim=cfgs.MODEL.z_dim,
        num_classes=cfgs.DATA.num_classes,
        y_sampler="acending_all",
        radius=cfgs.LOSS.radius,
        generator=Gen,
        discriminator=Dis,
        is_train=True,
        LOSS=cfgs.LOSS,
        RUN=cfgs.RUN,
        MODEL=cfgs.MODEL,
        device=local_rank,
        generator_mapping=None,
        generator_synthesis=None,
        is_stylegan=False,
        style_mixing_p="N/A",
        stylegan_update_emas=False,
        cal_trsp_cost=False) 

    print("===================================save pretrained images===================================")   

    misc.plot_img_canvas(images=fake_images.detach().cpu(),
                        save_path=join(cfgs.RUN.save_dir,
                        "figures/{run_name}/test_topk_ratio_{topk_ratio}_selectG_blocks_{selectG_blocks}_selectG_layers_{selectG_layers}.png".\
                        format(run_name = run_name, topk_ratio = cfgs.RUN.topk_ratio, selectG_blocks = cfgs.RUN.selectG_blocks, selectG_layers = cfgs.RUN.selectG_layers)),
                        num_cols=8,
                        logger=logger,
                        logging=global_rank == 0 and logger)
    


    print("===================================channel activation check===================================")

    # print(Gen.blocks[0][0].activation)
    # print(Gen.blocks[0][0])
    # print(Gen.blocks[1][0])
    # print(Gen.blocks[2][0])

    activations={}
    name = []
    length = 0 

    for key in Gen_reg_params.keys():
        name.append(key)
        print("Gen_reg_params{i}.shape".format(i=length), Gen_reg_params[key].size())
        length +=1 

    for i in range(length):
        Gen.blocks[i][0].deconv0.register_forward_hook(get_activation("{name}".format(name=name[i]), activations))

    # Gen.blocks[0][0].deconv0.register_forward_hook(get_activation("{name}".format(name=name[0]), activations))
    # Gen.blocks[1][0].deconv0.register_forward_hook(get_activation("{name}".format(name=name[1]), activations))
    # Gen.blocks[2][0].deconv0.register_forward_hook(get_activation("{name}".format(name=name[2]), activations))

    # Gen.blocks[0][0].activation.register_forward_hook(get_activation("{name}".format(name="block.0.0"), Gen_reg_params))
    # Gen.blocks[1][0].activation.register_forward_hook(get_activation("{name}".format(name="block.1.0"), Gen_reg_params))
    # Gen.blocks[2][0].activation.register_forward_hook(get_activation("{name}".format(name="block.2.0"), Gen_reg_params))


    # sample only target-class images
    for step_index in range(50000//cfgs.OPTIMIZATION.batch_size):

        fake_images, fake_labels, fake_images_eps, trsp_cost, ws, _, _ = sample.generate_images(
            z_prior=cfgs.MODEL.z_prior,
            truncation_factor=-1.0,
            batch_size=cfgs.OPTIMIZATION.batch_size,
            z_dim=cfgs.MODEL.z_dim,
            num_classes=cfgs.DATA.num_classes,
            y_sampler="forgetting",
            radius=cfgs.LOSS.radius,
            generator=Gen,
            discriminator=Dis,
            is_train=True,
            LOSS=cfgs.LOSS,
            RUN=cfgs.RUN,
            MODEL=cfgs.MODEL,
            device=local_rank,
            generator_mapping=None,
            generator_synthesis=None,
            is_stylegan=False,
            style_mixing_p="N/A",
            stylegan_update_emas=False,
            cal_trsp_cost=False) 
        

        for i in range(length):
            test = torch.mean(activations[name[i]], dim=0)
            test = test.view(test.size(dim=0),-1).mean(1)
            Gen_reg_params[name[i]] += test



        # test0 = torch.mean(activations[name[0]], dim=0)
        # test1 = torch.mean(activations[name[1]], dim=0)
        # test2 = torch.mean(activations[name[2]], dim=0)


        # test0 = test0.view(test0.size(dim=0),-1).mean(1)
        # test1 = test0.view(test1.size(dim=0),-1).mean(1)
        # test2 = test0.view(test2.size(dim=0),-1).mean(1)


        # Gen_reg_params[name[0]] += test0
        # Gen_reg_params[name[1]] += test1
        # Gen_reg_params[name[2]] += test2

    for i in range(length):
        print("block{i}.{i}".format(i=i),Gen_reg_params[name[i]].size())


    # Gen.blocks[0][0].deconv0.register_forward_hook(get_activation("{name}".format(name=name[0]), activations))
    # Gen.blocks[1][0].deconv0.register_forward_hook(get_activation("{name}".format(name=name[1]), activations))
    # Gen.blocks[2][0].deconv0.register_forward_hook(get_activation("{name}".format(name=name[2]), activations))

    # Gen.blocks[0][0].activation.register_forward_hook(get_activation("{name}".format(name="block.0.0"), Gen_reg_params))
    # Gen.blocks[1][0].activation.register_forward_hook(get_activation("{name}".format(name="block.1.0"), Gen_reg_params))
    # Gen.blocks[2][0].activation.register_forward_hook(get_activation("{name}".format(name="block.2.0"), Gen_reg_params))

    if dis_change == True:
        activations={}
        name = []
        length = 0 

        for key in Dis_reg_params.keys():
            name.append(key)
            length +=1 

        for i in range(length):
            Dis.blocks[i][0].conv0.register_forward_hook(get_activation("{name}".format(name=name[i]), activations))

        # sample only target-class images
        for step_index in range(50000//cfgs.OPTIMIZATION.batch_size):

            fake_images, fake_labels, fake_images_eps, trsp_cost, ws, _, _ = sample.generate_images(
                z_prior=cfgs.MODEL.z_prior,
                truncation_factor=-1.0,
                batch_size=cfgs.OPTIMIZATION.batch_size,
                z_dim=cfgs.MODEL.z_dim,
                num_classes=cfgs.DATA.num_classes,
                y_sampler="forgetting",
                radius=cfgs.LOSS.radius,
                generator=Gen,
                discriminator=Dis,
                is_train=True,
                LOSS=cfgs.LOSS,
                RUN=cfgs.RUN,
                MODEL=cfgs.MODEL,
                device=local_rank,
                generator_mapping=None,
                generator_synthesis=None,
                is_stylegan=False,
                style_mixing_p="N/A",
                stylegan_update_emas=False,
                cal_trsp_cost=False) 
            
            # calculate adv_output, embed, proxy, and cls_output using the discriminator
            fake_dict = Dis(fake_images, fake_labels, adc_fake=adc_fake)

            for i in range(length):
                test = torch.mean(activations[name[i]], dim=0)
                test = test.view(test.size(dim=0),-1).mean(1)
                Dis_reg_params[name[i]] += test



            # test0 = torch.mean(activations[name[0]], dim=0)
            # test1 = torch.mean(activations[name[1]], dim=0)
            # test2 = torch.mean(activations[name[2]], dim=0)


            # test0 = test0.view(test0.size(dim=0),-1).mean(1)
            # test1 = test0.view(test1.size(dim=0),-1).mean(1)
            # test2 = test0.view(test2.size(dim=0),-1).mean(1)


            # Gen_reg_params[name[0]] += test0
            # Gen_reg_params[name[1]] += test1
            # Gen_reg_params[name[2]] += test2

        for i in range(length):
            print("block{i}.{i}".format(i=i),Dis_reg_params[name[i]].size())

        
    # print("block0.0",Gen_reg_params[name[0]].size())
    # print("block1.0",Gen_reg_params[name[1]].size())
    # print("block2.0",Gen_reg_params[name[2]].size())

    sd_gen = Gen.state_dict()
    sd_dis = Dis.state_dict()
    print("==============================================================after step=======================================================")    

    if change == True:
        for name, _ in Gen.named_parameters():
            if name not in Gen_block:
                print("=========================================each layer==========================================")
                print("re initializing weights:", name)
                print("================before intersection================")
                print("topk_ratio:{percent}%".format(percent = 100/cfgs.RUN.topk_ratio))
                w = sd_gen[name].clone()
                size = sd_gen[name].size()
                print("re initializing weights_size", size)
                len = Gen_reg_params[name].size(dim=0)

                idx = torch.topk(Gen_reg_params[name], k= len//cfgs.RUN.topk_ratio)
                idx_sel = idx[1].detach().cpu().numpy()

                print("change idx", idx_sel)
                print("original filter", w[:, idx_sel[0]])
                print("original filter size", w[:, idx_sel[0]].size())

                weight_mask = torch.ones_like(w)
                print("before weight_mask", weight_mask[:,idx_sel[0]], weight_mask[:,idx_sel[0]+1])
                print("before weight_mask", weight_mask[:,idx_sel[0]].size(), weight_mask[:,idx_sel[0]+1].size())
                weight_mask[:, idx_sel] *= 0.0
                print("after weight_mask", weight_mask[:,idx_sel[0]], weight_mask[:,idx_sel[0]+1])
                print("after weight_mask", weight_mask[:,idx_sel[0]].size(), weight_mask[:,idx_sel[0]+1].size())

                new_weight = w * weight_mask
                sd_gen[name] = new_weight.data

                print("final filter", sd_gen[name][:, idx_sel[0]])
                print("final filter size", sd_gen[name][:, idx_sel[0]].size())


                # percent = (idx_sel.shape[0] / topk_size) * 100
                # print("turn_off ratio compared to topk_ratio: {percent}%".format(percent=percent))



        if dis_change == True:
            for name, _ in Dis.named_parameters():
                if name not in Dis_block:
                    print("=========================================each layer==========================================")
                    print("re initializing weights:", name)
                    print("================before intersection================")
                    print("topk_ratio:{percent}%".format(percent = 100/cfgs.RUN.topk_ratio))
                    w = sd_dis[name].clone()
                    size = sd_dis[name].size()
                    print("re initializing weights_size", size)
                    len = Dis_reg_params[name].size(dim=0)

                    idx = torch.topk(Dis_reg_params[name], k= len//cfgs.RUN.topk_ratio)
                    idx_sel = idx[1].detach().cpu().numpy()

                    print("change idx", idx_sel)
                    print("original filter", w[idx_sel[0]])
                    print("original filter size", w[idx_sel[0]].size())

                    weight_mask = torch.ones_like(w)
                    print("before weight_mask", weight_mask[idx_sel[0]], weight_mask[idx_sel[0]+1])
                    print("before weight_mask", weight_mask[idx_sel[0]].size(), weight_mask[idx_sel[0]+1].size())
                    weight_mask[idx_sel] *= 0.0
                    print("after weight_mask", weight_mask[idx_sel[0]], weight_mask[idx_sel[0]+1])
                    print("after weight_mask", weight_mask[idx_sel[0]].size(), weight_mask[idx_sel[0]+1].size())

                    new_weight = w * weight_mask
                    sd_dis[name] = new_weight.data

                    print("final filter", sd_dis[name][idx_sel[0]])
                    print("final filter size", sd_dis[name][idx_sel[0]].size())

    Gen.load_state_dict(sd_gen)
    Dis.load_state_dict(sd_dis)

    # print("==========================Dis load check==========================")
    # print(Dis.blocks[1][0].conv0.weight[idx_sel[0]])

    # print("==========================Dis load check==========================")
    # print(Dis.blocks[1][0].conv0.weight[idx_sel[0]])

    # "reset_seed"        
    # from torch.backends import cudnn
    # misc.fix_seed(cfgs.RUN.seed + global_rank)
    
    # # sample fake images and labels from p(G(z), y)
    # print("==================z_prior and fake_labels checking=================")
    # print("seed all",cfgs.RUN.seed + global_rank)
    # print("seed",cfgs.RUN.seed)
    # print("global_rank",global_rank)

    misc.make_GAN_untrainable(Gen, None, Dis)


    # sample only target-class images
    fake_images, fake_labels, fake_images_eps, trsp_cost, ws, _, _ = sample.generate_images(
        z_prior=cfgs.MODEL.z_prior,
        truncation_factor=-1.0,
        batch_size=cfgs.OPTIMIZATION.batch_size,
        z_dim=cfgs.MODEL.z_dim,
        num_classes=cfgs.DATA.num_classes,
        y_sampler="acending_all",
        radius=cfgs.LOSS.radius,
        generator=Gen,
        discriminator=Dis,
        is_train=True,
        LOSS=cfgs.LOSS,
        RUN=cfgs.RUN,
        MODEL=cfgs.MODEL,
        device=local_rank,
        generator_mapping=None,
        generator_synthesis=None,
        is_stylegan=False,
        style_mixing_p="N/A",
        stylegan_update_emas=False,
        cal_trsp_cost=False) 

    # print(fake_labels)

    misc.plot_img_canvas(images=fake_images.detach().cpu(),
                        save_path=join(cfgs.RUN.save_dir,
                        "figures/{run_name}/CHANNEL_no_intersection_topk_ratio_{topk_ratio}_selectG_blocks_{selectG_blocks}_selectG_layers_{selectG_layers}.png".\
                        format(run_name = run_name, topk_ratio = cfgs.RUN.topk_ratio, selectG_blocks = cfgs.RUN.selectG_blocks, selectG_layers = cfgs.RUN.selectG_layers)),
                        num_cols=8,
                        logger=logger,
                        logging=global_rank == 0 and logger)

    return Gen, Dis


def init_intersection(Gen_mas, Gen_block, Dis_mas, Dis_block, cfgs, global_rank, local_rank, logger, change=False):
    is_stylegan = cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]
    adc_fake = False

    from mas_utils.optimizer_lib import omega_update
    optimizer_ft_gen = omega_update(Gen_mas.source.parameters())
    misc.make_GAN_untrainable(Gen_mas.source, Dis_mas.source)

    # # toggle gradients of the generator and discriminator
    # misc.toggle_grad(model=self.Gen, grad=False, num_freeze_layers=-1, is_stylegan=self.is_stylegan)
    # misc.toggle_grad(model=self.Dis, grad=True, num_freeze_layers=self.RUN.freezeD, is_stylegan=self.is_stylegan)

    for step_index in range(50000//cfgs.OPTIMIZATION.batch_size):

        fake_images, fake_labels, _, _ = sample.generate_images(
            z_prior=cfgs.MODEL.z_prior,
            truncation_factor=-1.0,
            batch_size=cfgs.OPTIMIZATION.batch_size,
            z_dim=cfgs.MODEL.z_dim,
            num_classes=cfgs.DATA.num_classes,
            y_sampler="forgetting",
            radius=cfgs.LOSS.radius,
            generator=Gen_mas.source,
            discriminator=Dis_mas.source,
            is_train=True,
            LOSS=cfgs.LOSS,
            RUN=cfgs.RUN,
            MODEL=cfgs.MODEL,
            device=local_rank,
            is_stylegan=is_stylegan)

        # calculate adv_output, embed, proxy, and cls_output using the discriminator
        fake_dict = Dis_mas.source(fake_images_, fake_labels, adc_fake=adc_fake)

        
        "for_MAS"
        s= nn.Softmax(dim=1)

        cls_output_softmax = s(fake_dict["cls_output"])

        "disc.output=True"
        # if self.RUN.disc_output == True:
        #     sum_output = torch.sum(fake_dict["adv_output"])
        #     sum_output.backward()
        # else:
            # output = torch.max(fake_dict["cls_output"], dim=1)[0]
        output = torch.max(cls_output_softmax, dim=1)[0]
        sum_output = torch.sum(output)
        sum_output.backward()
        # print("check tensor:", check[1])
        # check_ = torch.zeros((self.OPTIMIZATION.batch_size, ), dtype=torch.long, device=device)

        optimizer_ft_gen.step(Gen_mas.reg_params, step_index, cfgs.OPTIMIZATION.batch_size, local_rank)
        # optimizer_ft_dis.step(self.Dis_mas.reg_params, 0, 1, self.local_rank)

    for step_index in range(50000//cfgs.OPTIMIZATION.batch_size):
        optimizer_ft_gen.zero_grad()

        fake_images, fake_labels, _, _ = sample.generate_images(
            z_prior=cfgs.MODEL.z_prior,
            truncation_factor=-1.0,
            batch_size=cfgs.OPTIMIZATION.batch_size,
            z_dim=cfgs.MODEL.z_dim,
            num_classes=cfgs.DATA.num_classes,
            y_sampler="forgetting_intersect",
            radius=cfgs.LOSS.radius,
            generator=Gen_mas.source,
            discriminator=Dis_mas.source,
            is_train=True,
            LOSS=cfgs.LOSS,
            RUN=cfgs.RUN,
            MODEL=cfgs.MODEL,
            device=local_rank,
            is_stylegan=is_stylegan)

        # apply differentiable augmentations if "apply_diffaug" or "apply_ada" is True
        fake_images_ = cfgs.AUG.series_augment(fake_images)

        # calculate adv_output, embed, proxy, and cls_output using the discriminator
        fake_dict = Dis_mas.source(fake_images_, fake_labels, adc_fake=adc_fake)

 
        "for_MAS"
        s= nn.Softmax(dim=1)

        cls_output_softmax = s(fake_dict["cls_output"])


        "disc.output=True"
        # if self.RUN.disc_output == True:
        #     sum_output = torch.sum(fake_dict["adv_output"])
        #     sum_output.backward()
        # else:
            # output = torch.max(fake_dict["cls_output"], dim=1)[0]
        output = torch.max(cls_output_softmax, dim=1)[0]
        sum_output = torch.sum(output)
        sum_output.backward()
        # print("check tensor:", check[1])
        # check_ = torch.zeros((self.OPTIMIZATION.batch_size, ), dtype=torch.long, device=device)

        optimizer_ft_gen.step(Gen_mas.reg_params_intersect, step_index, cfgs.OPTIMIZATION.batch_size, local_rank)
        # optimizer_ft_dis.step(self.Dis_mas.reg_params, 0, 1, self.local_rank)


    if change == True:
        sd_gen = Gen_mas.source.state_dict()
        print("after step=============================================================================")    
        # print(Gen_mas.reg_params)

        for name, _ in Gen_mas.source.named_parameters():
            if name not in Gen_block:
                for value in Gen_mas.reg_params.values():
                    if sd_gen[name].dim() == value["init_val"].dim():
                        if torch.equal(value["init_val"], sd_gen[name]):
                            value_clone = value["omega"].clone()

                for value in Gen_mas.reg_params_intersect.values():
                    if sd_gen[name].dim() == value["init_val"].dim():
                        if torch.equal(value["init_val"], sd_gen[name]):
                            value_clone_intersect = value["omega"].clone()

                #select index
                original_size = value_clone.size()
                w = torch.empty(original_size, device=local_rank)
                #zero masking
                torch.zeros_like(w)

                #re-init with xavierxavier
                # nn.init.xavier_uniform_(w)
                # nn.init.xavier_normal_(w)

                value_clone = value_clone.reshape(-1)
                value_clone_intersect = value_clone_intersect.reshape(-1)

                print("=========================================each layer==========================================")
                print("re initializing weights:", name)
                print(original_size)
                print("================before intersection================")
                print("value_clone",value_clone.shape)
                # print("value_clone_intersect",value_clone_intersect.shape)
                print("topk_ratio:{percent}%".format(percent = 100/cfgs.RUN.topk_ratio))
                size = list(value_clone.size())[0]
                idx = torch.topk(value_clone, k= size//cfgs.RUN.topk_ratio)
                idx_intersect = torch.topk(value_clone_intersect, k= int(size//cfgs.RUN.topk_ratio))

                topk_size = idx[1].shape[0]
                idx_sel = idx[1].detach().cpu().numpy()
                idx_intersect_sel = idx_intersect[1].detach().cpu().numpy()

                print("================how many index intersected????================")
                print("each of topk idx",idx_sel.shape)
                print("each of topk idx_intersect",idx_intersect_sel.shape)

                idx_sel.sort()
                idx_intersect_sel.sort()

                _, idx_intersect_sel_, _ = np.intersect1d(idx_sel,idx_intersect_sel, return_indices=True)
                print("=========after intersection=========")
                print("intersection weights size",idx_intersect_sel_.shape)

                idx_sel = torch.from_numpy(np.delete(idx_sel, idx_intersect_sel_)).to(local_rank)
                print("excluding intersection weights size",idx_sel.shape)

                percent = (idx_sel.shape[0] / topk_size) * 100
                print("turn_off ratio compared to topk_ratio: {percent}%".format(percent=percent))

                percent = (idx_intersect_sel_.shape[0] / topk_size) * 100
                print("intersection ratio compared to topk_ratio: {percent}%".format(percent=percent))

                param_clone = sd_gen[name].clone()
                param_clone = param_clone.reshape(-1)
                w = w.reshape(-1)

                param_clone[idx_sel] = w[idx_sel]
                print("===================before reshape==============")
                print("param_clone.shape",param_clone.shape)

                print("===================after reshape==============")
                param_clone = torch.reshape(param_clone, original_size)
                print("param_clone.shape",param_clone.shape)
                sd_gen[name] = param_clone.data

        Gen_mas.source.load_state_dict(sd_gen)

        # "reset_seed"        
        # from torch.backends import cudnn
        # misc.fix_seed(cfgs.RUN.seed + global_rank)
        
        # # sample fake images and labels from p(G(z), y)
        # print("==================z_prior and fake_labels checking=================")
        # print("seed all",cfgs.RUN.seed + global_rank)
        # print("seed",cfgs.RUN.seed)
        # print("global_rank",global_rank)

        misc.make_GAN_untrainable(Gen_mas.source, Dis_mas.source)


        # sample only target-class images
        fake_images, fake_labels, _, _ = sample.generate_images(
            z_prior=cfgs.MODEL.z_prior,
            truncation_factor=-1.0,
            batch_size=cfgs.OPTIMIZATION.batch_size,
            z_dim=cfgs.MODEL.z_dim,
            num_classes=cfgs.DATA.num_classes,
            y_sampler="acending_all",
            radius=cfgs.LOSS.radius,
            generator=Gen_mas.source,
            discriminator=Dis_mas.source,
            is_train=True,
            LOSS=cfgs.LOSS,
            RUN=cfgs.RUN,
            MODEL=cfgs.MODEL,
            device=local_rank,
            is_stylegan=is_stylegan)

        # print(fake_labels)

        misc.plot_img_canvas(images=fake_images.detach().cpu(),
                            save_path=join(cfgs.RUN.save_dir,
                            "figures/intersection_topk_{topk_ratio}_linear_{linear}_selectG_blocks_{selectG_blocks}_selectG_layers_{selectG_layers}.png".format(topk_ratio= cfgs.RUN.topk_ratio, linear = cfgs.RUN.linear, selectG_blocks = cfgs.RUN.selectG_blocks, selectG_layers = cfgs.RUN.selectG_layers)),
                            num_cols=8,
                            logger=logger,
                            logging=global_rank == 0 and logger)
    
    return Gen_mas

 
def init_no_intersection_channelwise(Gen_mas, Gen_block, Dis_mas, Dis_block, cfgs, global_rank, local_rank, logger, change = False, dis_change = False):
    
    is_stylegan = cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]
    adc_fake = False

    sd_gen = Gen_mas.source.state_dict()
    sd_dis = Dis_mas.source.state_dict()
    print("after step=============================================================================")    

    if dis_change == True:
        for name, _ in Dis_mas.source.named_parameters():
            if name not in Dis_block:
                for value in Dis_mas.reg_params.values():
                    if sd_dis[name].dim() == value["init_val"].dim():
                        if torch.equal(value["init_val"], sd_dis[name]):
                            value_clone = value["omega"].clone()


                print("==================================={name} masking=========================================".format(name=name))    

                #select index
                original_size = value_clone.size()
                w = sd_dis[name].clone()
                #zero masking
                w = unit_occupying(w, original_size, cfgs)
                sd_dis[name] = w

        Gen_mas.source.load_state_dict(sd_gen)
        Dis_mas.source.load_state_dict(sd_dis)

        # "reset_seed"        
        # from torch.backends import cudnn
        # misc.fix_seed(cfgs.RUN.seed + global_rank)
        
        # # sample fake images and labels from p(G(z), y)
        # print("==================z_prior and fake_labels checking=================")
        # print("seed all",cfgs.RUN.seed + global_rank)
        # print("seed",cfgs.RUN.seed)
        # print("global_rank",global_rank)

        misc.make_GAN_untrainable(Gen_mas.source, Dis_mas.source)


        # sample only target-class images
        fake_images, fake_labels, _, _ = sample.generate_images(
            z_prior=cfgs.MODEL.z_prior,
            truncation_factor=-1.0,
            batch_size=cfgs.OPTIMIZATION.batch_size,
            z_dim=cfgs.MODEL.z_dim,
            num_classes=cfgs.DATA.num_classes,
            y_sampler="acending_all",
            radius=cfgs.LOSS.radius,
            generator=Gen_mas.source,
            discriminator=Dis_mas.source,
            is_train=True,
            LOSS=cfgs.LOSS,
            RUN=cfgs.RUN,
            MODEL=cfgs.MODEL,
            device=local_rank,
            is_stylegan=is_stylegan)

        # print(fake_labels)

        misc.plot_img_canvas(images=fake_images.detach().cpu(),
                            save_path=join(cfgs.RUN.save_dir,
                            "figures/no_intersection_topk_ratio_{topk_ratio}_selectG_blocks_{selectG_blocks}_selectG_layers_{selectG_layers}.png".\
                            format(topk_ratio = cfgs.RUN.topk_ratio, selectG_blocks = cfgs.RUN.selectG_blocks, selectG_layers = cfgs.RUN.selectG_layers)),
                            num_cols=8,
                            logger=logger,
                            logging=global_rank == 0 and logger)
    
    return Gen_mas, Dis_mas


def unit_occupying(w, original_size, cfgs):
    out_channels = original_size[0]
    in_channels = original_size[1]
    minus_dim = 0
    randomly_select = True
    keepin = True
    keepout = True

    weight_mask = torch.ones_like(w)
    if keepin:
        new_in_channels = in_channels
    else:
        new_in_channels = int(cfgs.RUN.occupy_start * (in_channels - minus_dim)) + minus_dim

    if keepout:
        new_out_channels = out_channels
    else:
        new_out_channels = int(cfgs.RUN.occupy_start * out_channels)

    idx = np.arange(0, in_channels - minus_dim)
    if randomly_select:
        np.random.shuffle(idx)
    zero_idx = idx[new_in_channels-minus_dim:]
    weight_mask[:, zero_idx] *= 0.0

    idx = np.arange(0, out_channels)
    if randomly_select:
        np.random.shuffle(idx)
    zero_idx = idx[new_out_channels:]
    weight_mask[zero_idx, ] *= 0.0

    new_weight = w * weight_mask
    return new_weight