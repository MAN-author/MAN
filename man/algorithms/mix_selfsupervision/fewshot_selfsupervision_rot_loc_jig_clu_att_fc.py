import torch
import torch.nn.functional as F
import man.algorithms.algorithm as algorithm
import man.algorithms.classification.utils as cls_utils
import man.algorithms.fewshot.utils as fs_utils
import man.algorithms.mix_selfsupervision.rot_loc_jig_clu_utils_attention as rot_loc_jig_clu_utils_att
import man.utils as utils
import pdb
import time
from tqdm import tqdm
import numpy as np
from man.algorithms.mix_selfsupervision.gcn import GraphConvolution
class FewShotRotLocJigCluSelfSupervisionAttFC(algorithm.Algorithm):
    """Trains a few-shot model with the auxiliary location prediction task."""

    def __init__(self, opt, _run=None, _log=None, k = 10, w = 640):
        print(k)
        super().__init__(opt,_run, _log,k,w)
        self.keep_best_model_metric_name = "AccuracyNovel"
        self.rotation_loss_coef = opt["rotation_loss_coef"]
        self.patch_location_loss_coef = opt["patch_location_loss_coef"]
        self.patch_classification_loss_coef = opt["patch_classification_loss_coef"]
        self.cluster_loss_coef = opt["cluster_loss_coef"]
        self.clu_gcn_coef = opt["clu_gcn_coef"]
        self.standardize_image = opt["standardize_image"]
        self.combine_patches = opt["combine_patches"]
        self.standardize_patches = opt["standardize_patches"]
        self.rotation_invariant_classifier = opt["rotation_invariant_classifier"]
        self.random_rotation = opt["random_rotation"]
        if 'norm' in opt.keys():
            self.norm = opt["norm"]
        else:
            self.norm = None
        self.semi_supervised = opt["semi_supervised"] if ("semi_supervised" in opt) else False
        #self.lamda = torch.tensor([0.25, 0.25, 0.25, 0.25], requires_grad=True).cuda()
        #self.lamda_rot = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True).cuda()
        #self.lamda_loc = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True).cuda()
        #self.lamda_jig = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True).cuda()
        #self.lamda_clu = torch.nn.Parameter(torch.tensor(0.25), requires_grad=True).cuda()
        self.accuracies = {}

    def compute_features(self, dloader_unlabeled, epoch, feature_extractor, N):
        self.logger.info('Compute features')
        batch_time = utils.AverageMeter()
        end = time.time()
        # discard the label information in the dataloader
        for idx, batch in enumerate(tqdm(dloader_unlabeled(epoch))):
            assert len(batch) == 2
            (images_unlabeled,) = batch[1]
            images_unlabeled = self.tensors["images_unlabeled"].resize_(images_unlabeled.size()).copy_(
                images_unlabeled
            )
            features_batch = cls_utils.extract_features(
                feature_extractor, images_unlabeled
            )
            #pdb.set_trace()
            features_batch = features_batch.data.cpu().numpy()
            #pdb.set_trace()

            if idx == 0:
                features = np.zeros((N*features_batch.shape[0], 
                                     features_batch.shape[1], 
                                     features_batch.shape[2], 
                                     features_batch.shape[3]), 
                                    dtype='float32')

            features_batch = features_batch.astype('float32')
            if idx < len(dloader_unlabeled) - 1:
                features[idx * features_batch.shape[0]: (idx + 1) * features_batch.shape[0]] = features_batch
            else:
                # special treatment for final batch
                features[idx * features_batch.shape[0]:] = features_batch

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if  (idx % 200) == 0:
                self.logger.info('{0} / {1}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      .format(idx, len(dloader_unlabeled), batch_time=batch_time))
        return features
    
    def run_train_epoch(self, data_loader, epoch):
        #pdb.set_trace()
        self.logger.info(f"Training: {self.exp_name}")
        self.dloader = data_loader

        for key, network in self.networks.items():
            if self.optimizers[key] is None:
                network.eval()
            else:
                network.train()

        dname = ""
        if "name" in self.dloader.dataset.__dict__.keys():
            dname = self.dloader.dataset.name

        self.logger.info(f"==> Dataset: {dname} [{len(self.dloader)} batches]")
        disp_step = self.opt["disp_step"] if ("disp_step" in self.opt) else 50
        train_stats = utils.DAverageMeter("train", self._run)
        self.bnumber = len(data_loader)
        ##cluster begin
        # get the features for the whole dataset
        if self.cluster_loss_coef > 0.0:
            features = self.compute_features(data_loader,
                                    epoch,
                                    feature_extractor=self.networks["feature_extractor"],
                                    N=len(data_loader)
                                   )
            #print(features)
            features = features.reshape(features.shape[0], -1)
            features = torch.from_numpy(features)
            if self.clu_gcn_coef > 0.0:
                self.logger.info("gcn begin")
                
                #self.gcn.set_w(64)
                #self.gcn.set_w(512)
                features = self.gcn(features)
                
                if self.norm:
                    print('++++++++++++++++++++++++++++++++++++++++++++'+"norm"+'+++++++++++++++++++++++++++++++++++++')
                    features = F.normalize(features,p=2,dim=1)
                print(features)
                self.logger.info("gcn end")
            self.logger.info("fc begin")
            #clustering_loss = self.deepcluster.cluster(features, verbose=True)
            self.images_unlabeled_label = self.my_fc_cluster.run_cluster(features)
            self.logger.info("fc end")
            self.images_unlabeled_label = torch.Tensor(self.images_unlabeled_label).to(self.device).long()
            #print(self.images_unlabeled_label)
            #self.images_unlabeled_label = torch.Tensor(self.deepcluster.images_label).to(self.device).long()
            #pdb.set_trace()
        
        ##cluster end
        for idx, batch in enumerate(tqdm(data_loader(epoch))):
            #break
            #pdb.set_trace()
            self.biter = idx  # batch iteration.
            train_stats_this = self.train_step(batch)
            train_stats.update(train_stats_this)
            if (idx + 1) % disp_step == 0:
                self.logger.info(
                    "==> data_loader Iteration [%3d][%4d / %4d]: %s"
                    % (epoch + 1, idx + 1, len(data_loader), train_stats)
                )

        train_stats.log()
        self.add_stats_to_tensorboard_writer(train_stats.average(), "train_")

        return train_stats
    
        
    def allocate_tensors(self):
        self.tensors = {
            "images_train": torch.FloatTensor(),
            "labels_train": torch.LongTensor(),
            "labels_train_1hot": torch.FloatTensor(),
            "images_test": torch.FloatTensor(),
            "labels_test": torch.LongTensor(),
            "Kids": torch.LongTensor(),
            "patches_train": torch.FloatTensor(),
            "patches_test": torch.FloatTensor(),
            "patches": torch.FloatTensor(),
            "labels_patches": torch.LongTensor(),
            "images_unlabeled": torch.FloatTensor(),
            
        }

    def set_tensors(self, batch):
        two_datasets = (
            isinstance(batch, (list, tuple))
            and len(batch) == 2
            and isinstance(batch[0], (list, tuple))
            and isinstance(batch[1], (list, tuple))
        )

        if two_datasets:
            train_test_stage = "classification"
            assert len(batch[0]) == 5
            assert len(batch[1]) == 1
            images_test, patches_test, labels_test, K, num_base_per_episode = batch[0]
            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)
            patches_test = patches_test.view(
                patches_test.size(0) * patches_test.size(1),  # 1 * 64
                patches_test.size(2),  # 9
                patches_test.size(3),  # 3
                patches_test.size(4),  # 24
                patches_test.size(5),
            )  # 24
            labels_patches = labels_test.view(-1)
            #pdb.set_trace()
            self.tensors["patches"].resize_(patches_test.size()).copy_(patches_test)
            self.tensors["labels_patches"].resize_(labels_patches.size()).copy_(labels_patches)
            (images_unlabeled,) = batch[1]
            images_unlabeled = self.tensors["images_unlabeled"].resize_(images_unlabeled.size()).copy_(
                images_unlabeled
            )

        elif len(batch) == 6:
            train_test_stage = "fewshot"
            (
                images_train,
                labels_train,
                images_test,
                labels_test,
                K,
                num_base_per_episode,
            ) = batch
            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_train"].resize_(images_train.size()).copy_(images_train)
            self.tensors["labels_train"].resize_(labels_train.size()).copy_(labels_train)
            labels_train = self.tensors["labels_train"]

            nKnovel = 1 + labels_train.max().item() - self.num_base

            labels_train_1hot_size = list(labels_train.size()) + [
                nKnovel,
            ]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            self.tensors["labels_train_1hot"].resize_(labels_train_1hot_size).fill_(
                0
            ).scatter_(
                len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.num_base, 1
            )
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)

        elif len(batch) == 8:
            train_test_stage = "fewshot"
            (
                images_train,
                patches_train,
                labels_train,
                images_test,
                patches_test,
                labels_test,
                K,
                num_base_per_episode,
            ) = batch

            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_train"].resize_(images_train.size()).copy_(images_train)
            self.tensors["patches_train"].resize_(patches_train.size()).copy_(patches_train)
            self.tensors["labels_train"].resize_(labels_train.size()).copy_(labels_train)
            labels_train = self.tensors["labels_train"]

            nKnovel = 1 + labels_train.max().item() - self.num_base

            labels_train_1hot_size = list(labels_train.size()) + [
                nKnovel,
            ]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            self.tensors["labels_train_1hot"].resize_(labels_train_1hot_size).fill_(
                0
            ).scatter_(
                len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.num_base, 1
            )
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["patches_test"].resize_(patches_test.size()).copy_(patches_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)

        elif len(batch) == 4:
            train_test_stage = "classification"
            images_test, labels_test, K, num_base_per_episode = batch

            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)
        elif len(batch) == 5:
            train_test_stage = "classification"
            images_test, patches_test, labels_test, K, num_base_per_episode = batch

            self.num_base = num_base_per_episode[0].item()
            self.tensors["images_test"].resize_(images_test.size()).copy_(images_test)
            self.tensors["labels_test"].resize_(labels_test.size()).copy_(labels_test)
            self.tensors["Kids"].resize_(K.size()).copy_(K)

            patches_test = patches_test.view(
                patches_test.size(0) * patches_test.size(1),  # 1 * 64
                patches_test.size(2),  # 9
                patches_test.size(3),  # 3
                patches_test.size(4),  # 24
                patches_test.size(5),
            )  # 24
            labels_patches = labels_test.view(-1)
            #pdb.set_trace()
            self.tensors["patches"].resize_(patches_test.size()).copy_(patches_test)
            self.tensors["labels_patches"].resize_(labels_patches.size()).copy_(labels_patches)

        return train_test_stage

    def train_step(self, batch):
        return self.process_batch(batch, is_train=True)

    def evaluation_step(self, batch):
        #pdb.set_trace()
        return self.process_batch(batch, is_train=False)

    def process_batch(self, batch, is_train):
        process_type = self.set_tensors(batch)
        #pdb.set_trace()
        if process_type == "fewshot":
            record = self.process_batch_fewshot_classification_task(is_train)
        elif process_type == "classification":
            record = self.process_batch_base_class_classification_task(is_train)
        else:
            raise ValueError(f"Unexpected process type {process_type}")

        return record

    def process_batch_base_class_classification_task(self, is_train):

        images = self.tensors["images_test"]
        labels = self.tensors["labels_test"]
        Kids = self.tensors["Kids"]
        base_ids = Kids[:, : self.num_base].contiguous()
        assert images.dim() == 5 and labels.dim() == 2
        images = utils.convert_from_5d_to_4d(images)

        if self.standardize_image:
            images = utils.standardize_image(images)

        labels = labels.view(-1)

        patches = self.tensors["patches"]
        labels_patches = self.tensors["labels_patches"]

        auxiliary_tasks = is_train and (
            self.patch_location_loss_coef > 0.0 or self.patch_classification_loss_coef > 0.0 or self.cluster_loss_coef > 0.0 or \
            self.rotation_loss_coef > 0.0
        )
        if self.semi_supervised and is_train:
            images_unlabeled = self.tensors["images_unlabeled"]
            images_unlabeled_label_this=self.images_unlabeled_label[self.biter*images_unlabeled.shape[0]:(self.biter+1)*images_unlabeled.shape[0]]
            assert images_unlabeled.dim() == 4
        else:
            images_unlabeled = None
            images_unlabeled_label_this = None
        
        assert auxiliary_tasks == True

        if auxiliary_tasks:
            #record = rot_loc_jig_clu_utils.object_classification_with_rot_loc_jig_clu_selfsupervision(
            #'''
            record = rot_loc_jig_clu_utils_att.object_classification_with_rot_loc_jig_clu_selfsupervision(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers["feature_extractor"],
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers["classifier"],
                #attention
                #att_classifier=self.networks.get("classifier_att"),
                #att_classifier_optimizer=self.optimizers.get("classifier_att"),
                #lamda = self.lamda,
                #attention
                rot_classifier=self.networks.get("classifier_rot"),
                rot_classifier_optimizer=self.optimizers.get("classifier_rot"),
                location_classifier=self.networks.get("classifier_loc"),
                location_classifier_optimizer=self.optimizers.get("classifier_loc"),
                jig_classifier=self.networks.get("classifier_jig"),
                jig_classifier_optimizer=self.optimizers.get("classifier_jig"),
                clu_classifier=self.networks.get("classifier_clu"),
                clu_classifier_optimizer=self.optimizers.get("classifier_clu"),
                patch_classifier=self.networks.get("patch_classifier"),
                patch_classifier_optimizer=self.optimizers.get("patch_classifier"),
                images=images,
                labels=labels,
                patches=patches,
                labels_patches=labels_patches,
                is_train=is_train,
                rotation_loss_coef=self.rotation_loss_coef,
                patch_location_loss_coef=self.patch_location_loss_coef,
                patch_classification_loss_coef=self.patch_classification_loss_coef,
                cluster_loss_coef=self.cluster_loss_coef,
                random_rotation=self.random_rotation,
                rotation_invariant_classifier=self.rotation_invariant_classifier,
                combine=self.combine_patches,
                base_ids=base_ids,
                standardize_patches=self.standardize_patches,
                images_unlabeled=images_unlabeled,
                images_unlabeled_label=images_unlabeled_label_this
            )
            '''
            feature_extractor=self.networks["feature_extractor"]
            feature_extractor_optimizer=self.optimizers["feature_extractor"]
            classifier=self.networks["classifier"]
            classifier_optimizer=self.optimizers["classifier"]
            rot_classifier=self.networks.get("classifier_rot")
            rot_classifier_optimizer=self.optimizers.get("classifier_rot")
            location_classifier=self.networks.get("classifier_loc")
            location_classifier_optimizer=self.optimizers.get("classifier_loc")
            jig_classifier=self.networks.get("classifier_jig")
            jig_classifier_optimizer=self.optimizers.get("classifier_jig")
            clu_classifier=self.networks.get("classifier_clu")
            clu_classifier_optimizer=self.optimizers.get("classifier_clu")
            patch_classifier=self.networks.get("patch_classifier")
            patch_classifier_optimizer=self.optimizers.get("patch_classifier")
            if base_ids is not None:
                assert base_ids.size(0) == 1

            assert images.dim() == 4
            assert labels.dim() == 1
            assert images.size(0) == labels.size(0)
            #assert att_classifier != None
            #assert att_classifier_optimizer != None
            assert patches.dim() == 5 and patches.size(1) == 9
            assert patches.size(0) == labels_patches.size(0)
            patches = utils.convert_from_5d_to_4d(patches)
            if self.standardize_patches:
                patches = utils.standardize_image(patches)

            if is_train:
                # Zero gradients.
                feature_extractor_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                #if att_classifier_optimizer != None:
                #    att_classifier_optimizer.zero_grad()
                if self.rotation_loss_coef > 0.0:
                    rot_classifier_optimizer.zero_grad()
                if self.patch_location_loss_coef > 0.0:
                    location_classifier_optimizer.zero_grad()
                    jig_classifier_optimizer.zero_grad()
                if self.cluster_loss_coef > 0.0:
                    clu_classifier_optimizer.zero_grad()
                if self.patch_classification_loss_coef > 0.0:
                    patch_classifier_optimizer.zero_grad()
            with torch.no_grad():
                images, labels, labels_rotation = rot_loc_jig_clu_utils_att.preprocess_input_data(
                    images, labels, None, self.random_rotation, self.rotation_invariant_classifier
                )
            record = {}
            with torch.set_grad_enabled(is_train):
                # Extract features from the images.
                features_images = feature_extractor(images)
                # Extract features from the image patches.
                features_patches = feature_extractor(patches)
                #pdb.set_trace()
                #pdb.set_trace()
                # Perform object classification task.
                scores_classification, loss_classsification = cls_utils.classification_task(
                    classifier, features_images, labels, base_ids
                )
                record["loss_cls"] = loss_classsification.item()
                loss_total = loss_classsification
        
                #attention 
                #self.lamda_rot = self.lamda[0] / torch.sum(self.lamda)
                #self.lamda_loc = self.lamda[1] / torch.sum(self.lamda)
                #self.lamda_jig = self.lamda[2] / torch.sum(self.lamda)
                #self.lamda_clu = self.lamda[3] / torch.sum(self.lamda)
                #lamda_rot,lamda_loc, lamda_jig,lamda_clu = att_classifier(features_images).mean(0)
                record["lamda_rot"] = self.lamda_rot.item()
                record["lamda_loc"] = self.lamda_loc.item()
                record["lamda_jig"] = self.lamda_jig.item()
                record["lamda_clu"] = self.lamda_clu.item()
                #print(self.lamda_rot, self.lamda_loc, self.lamda_jig, self.lamda_clu)

                if self.rotation_loss_coef > 0.0:
                    scores_rotation, loss_rotation =  rot_loc_jig_clu_utils_att.rotation_task(
                        rot_classifier, features_images, labels_rotation
                    )
                    record["loss_rot"] = loss_rotation.item()
                    loss_total = loss_total + loss_rotation * self.lamda_rot
                    
                if self.patch_location_loss_coef > 0.0:
                    # patch location prediction.
                    scores_location, loss_location, labels_loc =  rot_loc_jig_clu_utils_att.patch_location_task(
                        location_classifier, features_patches
                    )
                    record["loss_loc"] = loss_location.item()
                    loss_total = loss_total + loss_location * self.lamda_loc
            
                    # patch puzzle prediction.
                    scores_puzzle, loss_puzzle, labels_puzzle =  rot_loc_jig_clu_utils_att.jigsaw_puzzle_task(
                        jig_classifier, features_patches
                    )
                    record["loss_jig"] = loss_puzzle.item()
                    loss_total = loss_total + loss_puzzle * self.lamda_jig
                if self.cluster_loss_coef > 0.0:
                    #cluster prediction.
                    features_unlabeled = cls_utils.extract_features(
                        feature_extractor, images_unlabeled
                    )
                    scores_cluster, loss_cluster =  rot_loc_jig_clu_utils_att.cluster_task(
                        clu_classifier, features_unlabeled, images_unlabeled_label_this
                    )
                    record["loss_clu"] = loss_cluster.item()
                    loss_total = loss_total + loss_cluster * self.lamda_clu
                #pdb.set_trace()
                # Perform the auxiliary task of classifying individual patches.
                if self.patch_classification_loss_coef > 0.0:
                    scores_patch, loss_patch = rot_loc_jig_clu_utils_att.patch_classification_task(
                        patch_classifier, features_patches, labels_patches, self.combine_patches
                    )
                    record["loss_patch_cls"] = loss_patch.item()
                    loss_total = loss_total + loss_patch * self.patch_classification_loss_coef

                # Because the total loss consists of multiple individual losses
                # (i.e., 3) scale it down by a factor of 0.5.
                loss_total = loss_total * 0.5

            with torch.no_grad():
                # Compute accuracies.
                record["Accuracy"] = utils.top1accuracy(scores_classification, labels)
                if self.rotation_loss_coef > 0.0:
                    record["AccuracyRot"] = utils.top1accuracy(scores_rotation, labels_rotation)
                if self.patch_location_loss_coef > 0.0:
                    record["AccuracyLoc"] = utils.top1accuracy(scores_location, labels_loc)
                    record["AccuracyJig"] = utils.top1accuracy(scores_puzzle, labels_puzzle)
                if self.cluster_loss_coef > 0.0:
                    record["AccuracyClu"] = utils.top1accuracy(scores_cluster, images_unlabeled_label_this)
                if self.patch_classification_loss_coef > 0.0:
                    record["AccuracyPatch"] = utils.top1accuracy(scores_patch, labels_patches)

            if is_train:
                # Backward loss and apply gradient steps.
                loss_total.backward()
                feature_extractor_optimizer.step()
                classifier_optimizer.step()
                #att_classifier_optimizer.step()
                if self.rotation_loss_coef > 0.0:
                    rot_classifier_optimizer.step()
                if self.patch_location_loss_coef > 0.0:
                    location_classifier_optimizer.step()
                    jig_classifier_optimizer.step()
                if self.cluster_loss_coef > 0.0:
                    clu_classifier_optimizer.step()
                if self.patch_classification_loss_coef > 0.0:
                    patch_classifier_optimizer.step()
            #pdb.set_trace()
            '''
        else:
            record = cls_utils.object_classification(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers["feature_extractor"],
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers["classifier"],
                images=images,
                labels=labels,
                is_train=is_train,
                base_ids=base_ids,
            )

        return record

    def process_batch_fewshot_classification_task(self, is_train):
        Kids = self.tensors["Kids"]
        base_ids = None if (self.num_base == 0) else Kids[:, : self.num_base].contiguous()

        images_train = self.tensors["images_train"]
        images_test = self.tensors["images_test"]

        if self.standardize_image:
            assert images_train.dim() == 5 and images_test.dim() == 5
            assert images_train.size(0) == images_test.size(0)
            meta_batch_size = images_train.size(0)
            images_train = utils.convert_from_5d_to_4d(images_train)
            images_test = utils.convert_from_5d_to_4d(images_test)

            images_train = utils.standardize_image(images_train)
            images_test = utils.standardize_image(images_test)

            images_train = utils.add_dimension(images_train, meta_batch_size)
            images_test = utils.add_dimension(images_test, meta_batch_size)

        auxiliary_tasks = is_train and (
            self.patch_location_loss_coef > 0.0 or self.patch_classification_loss_coef > 0.0
        )
        assert auxiliary_tasks == False
        if auxiliary_tasks:
            record = jpu_utils.fewshot_classification_with_jigsaw_puzzle_selfsupervision(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers["feature_extractor"],
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers["classifier"],
                location_classifier=self.networks.get("classifier_jig"),
                location_classifier_optimizer=self.optimizers.get("classifier_jig"),
                patch_classifier=self.networks.get("patch_classifier"),
                patch_classifier_optimizer=self.optimizers.get("patch_classifier"),
                images_train=images_train,
                patches_train=self.tensors["patches_train"],
                labels_train=self.tensors["labels_train"],
                labels_train_1hot=self.tensors["labels_train_1hot"],
                images_test=images_test,
                patches_test=self.tensors["patches_test"],
                labels_test=self.tensors["labels_test"],
                is_train=is_train,
                base_ids=base_ids,
                patch_location_loss_coef=self.patch_location_loss_coef,
                patch_classification_loss_coef=self.patch_classification_loss_coef,
                combine=self.combine_patches,
                standardize_patches=self.standardize_patches,
            )
        else:
            record = fs_utils.fewshot_classification(
                feature_extractor=self.networks["feature_extractor"],
                feature_extractor_optimizer=self.optimizers.get("feature_extractor"),
                classifier=self.networks["classifier"],
                classifier_optimizer=self.optimizers.get("classifier"),
                images_train=images_train,
                labels_train=self.tensors["labels_train"],
                labels_train_1hot=self.tensors["labels_train_1hot"],
                images_test=images_test,
                labels_test=self.tensors["labels_test"],
                is_train=is_train,
                base_ids=base_ids,
            )

        if not is_train:
            record, self.accuracies = fs_utils.compute_95confidence_intervals(
                record,
                episode=self.biter,
                num_episodes=self.bnumber,
                store_accuracies=self.accuracies,
                metrics=["AccuracyNovel",],
            )

        return record
