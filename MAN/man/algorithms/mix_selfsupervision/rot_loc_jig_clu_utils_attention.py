import torch
import torch.nn.functional as F
import pdb
import man.algorithms.classification.utils as cls_utils
import man.algorithms.fewshot.utils as fewshot_utils
import man.utils as utils
import numpy as np

_CENTRAL_PATCH_INDEX = 4
_NUM_OF_PATCHES = 9
_NUM_LOCATION_CLASSES = _NUM_OF_PATCHES - 1
_NUM_PUZZLE_CLASSES = 64

### rotation begin###
def rotation_task(rotation_classifier, features, labels_rotation):
    """Applies the rotation prediction head to the given features."""
    scores, loss = rotation_classifier(features,labels_rotation)
    assert scores.size(1) == 4
    #loss = F.cross_entropy(scores, labels_rotation)

    return scores, loss


def create_rotations_labels(batch_size, device):
    """Creates the rotation labels."""
    labels_rot = torch.arange(4, device=device).view(4, 1)

    labels_rot = labels_rot.repeat(1, batch_size).view(-1)
    return labels_rot


def create_4rotations_images(images, stack_dim=None):
    """Rotates each image in the batch by 0, 90, 180, and 270 degrees."""
    images_4rot = []
    for r in range(4):
        images_4rot.append(utils.apply_2d_rotation(images, rotation=r * 90))

    if stack_dim is None:
        images_4rot = torch.cat(images_4rot, dim=0)
    else:
        images_4rot = torch.stack(images_4rot, dim=stack_dim)

    return images_4rot


def randomly_rotate_images(images):
    """Randomly rotates each image in the batch by 0, 90, 180, or 270 degrees."""
    batch_size = images.size(0)
    labels_rot = torch.from_numpy(np.random.randint(0, 4, size=batch_size))
    labels_rot = labels_rot.to(images.device)

    for r in range(4):
        mask = labels_rot == r
        images_masked = images[mask].contiguous()
        images[mask] = utils.apply_2d_rotation(images_masked, rotation=r * 90)

    return images, labels_rot


def extend_with_unlabeled_images_for_rotation(images_unlabeled, images, labels_rotation):
    """Extend a mini-batch with unlabeled images for the rotation task."""
    batch_size_in_unlabeled = images_unlabeled.size(0)
    images_unlabeled = create_4rotations_images(images_unlabeled)
    labels_unlabeled_rotation = create_rotations_labels(
        batch_size_in_unlabeled, images_unlabeled.device
    )

    images = torch.cat([images, images_unlabeled], dim=0)
    labels_rotation = torch.cat([labels_rotation, labels_unlabeled_rotation], dim=0)

    return images, labels_rotation


def preprocess_input_data(
    images,
    labels=None,
    images_unlabeled=None,
    random_rotation=False,
    rotation_invariant_classifier=None,
):
    """Preprocess a mini-batch of images."""

    if labels is not None:
        assert rotation_invariant_classifier is not None

    if random_rotation:
        assert (labels is None) or (rotation_invariant_classifier is True)
        # randomly rotate an image.
        assert images_unlabeled is None
        if images_unlabeled is not None:
            images = torch.cat([images, images_unlabeled], dim=0)
        images, labels_rotation = randomly_rotate_images(images)
    else:
        # Create the 4 rotated version of the images; this step increases
        # the batch size by a multiple of 4.
        batch_size_in = images.size(0)
        images = create_4rotations_images(images)
        labels_rotation = create_rotations_labels(batch_size_in, images.device)

        if images_unlabeled is not None:
            images, labels_rotation = extend_with_unlabeled_images_for_rotation(
                images_unlabeled, images, labels_rotation
            )

        if (labels is not None) and (rotation_invariant_classifier is True):
            # Extend labels so as to train a rotation invariant object
            # classifier, i.e., all rotated versions of an image use the
            # same label.
            labels = labels.repeat(4)

    return images, labels, labels_rotation

###rotation end###


### utility  of two selfsupervision method ### 
def add_patch_dimension(patches):
    """Add the patch dimension to a mini-batch of patches."""
    assert (patches.size(0) % _NUM_OF_PATCHES) == 0
    return utils.add_dimension(patches, dim_size=(patches.size(0) // _NUM_OF_PATCHES))

### method of patch classification ### 
def concatenate_accross_channels_patches(patches):
    assert patches.dim() == 3 or patches.dim() == 5
    if patches.dim() == 3:
        batch_size, _, channels = patches.size()
        return patches.view(batch_size, _NUM_OF_PATCHES * channels)
    elif patches.dim() == 5:
        batch_size, _, channels, height, width = patches.size()
        return patches.view(batch_size, _NUM_OF_PATCHES * channels, height, width)

### method of patch classification ### 
def combine_multiple_patch_features(features, combine):
    """Combines the multiple patches of an image."""
    if combine == "average":
        return features.mean(dim=1)
    elif combine == "concatenate":
        return concatenate_accross_channels_patches(features)
    else:
        raise ValueError(f"Not supported combine option {combine}")

    
### method of patch locations###
def generate_patch_locations():
    """Generates patch locations."""
    locations = [i for i in range(_NUM_OF_PATCHES) if i != _CENTRAL_PATCH_INDEX]
    assert len(locations) == _NUM_LOCATION_CLASSES
    return _CENTRAL_PATCH_INDEX, locations

### method of patch locations###
def generate_location_labels(batch_size, is_cuda):
    """Generates location prediction labels."""
    location_labels = torch.arange(_NUM_LOCATION_CLASSES).view(1, _NUM_LOCATION_CLASSES)
    if is_cuda:
        location_labels = location_labels.to("cuda")
    location_labels = location_labels.repeat(batch_size, 1).view(-1)
    return location_labels

### method of patch locations###
def create_patch_pairs(patches, central, locations):
    """Creates patch pairs."""
    assert patches.size(1) == 9
    num_dims = patches.dim()
    if num_dims == 3:
        patches = patches.view(patches.size(0), 9, patches.size(2), 1, 1)

    assert patches.dim() == 5
    batch_size, _, channels, height, width = patches.size()

    patches_central = patches[:, central, :, :, :]
    patch_pairs = []
    for loc in locations:
        patches_loc = patches[:, loc, :, :, :]
        patch_pairs.append(torch.cat([patches_loc, patches_central], dim=1))

    patch_pairs = torch.stack(patch_pairs, dim=1)
    if num_dims == 3:
        patch_pairs = patch_pairs.view(batch_size * len(locations), 2 * channels)
    else:
        patch_pairs = patch_pairs.view(
            batch_size * len(locations), 2 * channels, height, width
        )

    return patch_pairs


### method of patch puzzle###
def generate_jig_labels(batch_size, is_cuda):
    """Generates location prediction labels."""
    jig_labels = torch.arange(_NUM_PUZZLE_CLASSES).view(1, _NUM_PUZZLE_CLASSES)
    if is_cuda:
        jig_labels = jig_labels.to("cuda")
    jig_labels = jig_labels.repeat(batch_size, 1).view(-1)
    return jig_labels


### method of patch puzzle###
def create_patch_puzzle(patches):
    """Creates patch puzzle."""
    assert patches.size(1) == 9
    num_dims = patches.dim()
    if num_dims == 3:
        patches = patches.view(patches.size(0), 9, patches.size(2), 1, 1)

    assert patches.dim() == 5
    batch_size, _, channels, height, width = patches.size()

    patch_puzzle = []
    puzzle = np.load("man/algorithms/selfsupervision/permutations_hamming_max_64.npy")
    for nclasses in range(puzzle.shape[0]):
        for loc in range(puzzle.shape[1]):
            patches_loc = patches[:, puzzle[nclasses][loc], :, :, :]
            patch_puzzle.append(patches_loc)

    patch_puzzle = torch.stack(patch_puzzle, dim=1)
    if num_dims == 3:
        patch_puzzle = patch_puzzle.view(batch_size * puzzle.shape[0], puzzle.shape[1] * channels)
    else:
        patch_puzzle = patch_puzzle.view(
            batch_size * puzzle.shape[0], puzzle.shape[1] * channels, height, width
        )

    return patch_puzzle

### tasks begin ###
def patch_location_task(location_classifier, features):
    """Applies the patch location prediction head to the given features."""
    features = add_patch_dimension(features)
    batch_size = features.size(0)

    central, locations = generate_patch_locations()
    location_labels = generate_location_labels(batch_size, features.is_cuda)
    features_pairs = create_patch_pairs(features, central, locations)
    scores,loss = location_classifier(features_pairs,location_labels)
    assert scores.size(1) == _NUM_LOCATION_CLASSES
    #loss = F.cross_entropy(scores, location_labels)
    #pdb.set_trace()
    return scores, loss, location_labels


def jigsaw_puzzle_task(jig_classifier, features):
    """Applies the patch location prediction head to the given features."""
    features = add_patch_dimension(features)
    batch_size = features.size(0)

    jig_labels = generate_jig_labels(batch_size, features.is_cuda)
    features_puzzle = create_patch_puzzle(features)
    scores,loss = jig_classifier(features_puzzle, jig_labels)
    assert scores.size(1) == _NUM_PUZZLE_CLASSES
    #loss = F.cross_entropy(scores, jig_labels)
    #pdb.set_trace()
    return scores, loss, jig_labels

def cluster_task(cluster_classifier, features, labels):
    """Applies the rotation prediction head to the given features."""
    scores, loss = cluster_classifier(features,labels)
    #pdb.set_trace()
    #labels = torch.Tensor(labels).cuda()
    #assert scores.size(1) == 10
    #loss = F.cross_entropy(scores, labels)
    

    return scores, loss


def patch_classification_task(patch_classifier, features, labels, combine):
    """Applies the auxiliary task of classifying individual patches."""
    features = add_patch_dimension(features)
    features = combine_multiple_patch_features(features, combine)

    #scores,loss = patch_classifier(features,labels=labels)
    scores = patch_classifier(features)
    loss = F.cross_entropy(scores, labels)
    return scores, loss

### tasks end ###

def object_classification_with_rot_loc_jig_clu_selfsupervision(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    #att_classifier,
    #att_classifier_optimizer,
    #lamda,
    rot_classifier,
    rot_classifier_optimizer,
    location_classifier,
    location_classifier_optimizer,
    jig_classifier,
    jig_classifier_optimizer,
    clu_classifier,
    clu_classifier_optimizer,
    patch_classifier,
    patch_classifier_optimizer,
    images,
    labels,
    patches,
    labels_patches,
    is_train,
    rotation_loss_coef=1.0,
    patch_location_loss_coef=1.0,
    patch_classification_loss_coef=1.0,
    cluster_loss_coef=1.0,
    random_rotation=False,
    rotation_invariant_classifier=False,
    combine="average",
    base_ids=None,
    standardize_patches=True,
    images_unlabeled=None,
    images_unlabeled_label=None
):
    """Forward-backward propagation routine for classification model extended
    with the auxiliary self-supervised task of predicting the relative location
    of patches."""

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
    if standardize_patches:
        patches = utils.standardize_image(patches)

    if is_train:
        # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        #if att_classifier_optimizer != None:
        #    att_classifier_optimizer.zero_grad()
        if rotation_loss_coef > 0.0:
            rot_classifier_optimizer.zero_grad()
        if patch_location_loss_coef > 0.0:
            location_classifier_optimizer.zero_grad()
            jig_classifier_optimizer.zero_grad()
        if cluster_loss_coef > 0.0:
            clu_classifier_optimizer.zero_grad()
        if patch_classification_loss_coef > 0.0:
            patch_classifier_optimizer.zero_grad()
    with torch.no_grad():
        images, labels, labels_rotation = preprocess_input_data(
            images, labels, None, random_rotation, rotation_invariant_classifier
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
        #lamda_rot = lamda[0] / torch.sum(lamda)
        #lamda_loc = lamda[1] / torch.sum(lamda)
        #lamda_jig = lamda[2] / torch.sum(lamda)
        #lamda_clu = lamda[3] / torch.sum(lamda)
        #lamda_rot,lamda_loc, lamda_jig,lamda_clu = att_classifier(features_images).mean(0)
        #record["lamda_rot"] = lamda_rot.item()
        #record["lamda_loc"] = lamda_loc.item()
        #record["lamda_jig"] = lamda_jig.item()
        #record["lamda_clu"] = lamda_clu.item()
        #print(lamda)

        if rotation_loss_coef > 0.0:
            scores_rotation, loss_rotation = rotation_task(
                rot_classifier, features_images, labels_rotation
            )
            record["loss_rot"] = loss_rotation.item()
            loss_total = loss_total + loss_rotation 
            
        if patch_location_loss_coef > 0.0:
            # patch location prediction.
            scores_location, loss_location, labels_loc = patch_location_task(
                location_classifier, features_patches
            )
            record["loss_loc"] = loss_location.item()
            loss_total = loss_total + loss_location 
            
            # patch puzzle prediction.
            scores_puzzle, loss_puzzle, labels_puzzle = jigsaw_puzzle_task(
                jig_classifier, features_patches
            )
            record["loss_jig"] = loss_puzzle.item()
            loss_total = loss_total + loss_puzzle 
        if cluster_loss_coef > 0.0:
            #cluster prediction.
            features_unlabeled = cls_utils.extract_features(
                feature_extractor, images_unlabeled
            )
            scores_cluster, loss_cluster = cluster_task(
                clu_classifier, features_unlabeled, images_unlabeled_label
            )
            record["loss_clu"] = loss_cluster.item()
            loss_total = loss_total + loss_cluster
        #pdb.set_trace()
        # Perform the auxiliary task of classifying individual patches.
        if patch_classification_loss_coef > 0.0:
            scores_patch, loss_patch = patch_classification_task(
                patch_classifier, features_patches, labels_patches, combine
            )
            record["loss_patch_cls"] = loss_patch.item()
            loss_total = loss_total + loss_patch * patch_classification_loss_coef

        # Because the total loss consists of multiple individual losses
        # (i.e., 3) scale it down by a factor of 0.5.
        loss_total = loss_total * 0.5

    with torch.no_grad():
        # Compute accuracies.
        record["Accuracy"] = utils.top1accuracy(scores_classification, labels)
        if rotation_loss_coef > 0.0:
            record["AccuracyRot"] = utils.top1accuracy(scores_rotation, labels_rotation)
        if patch_location_loss_coef > 0.0:
            record["AccuracyLoc"] = utils.top1accuracy(scores_location, labels_loc)
            record["AccuracyJig"] = utils.top1accuracy(scores_puzzle, labels_puzzle)
        if cluster_loss_coef > 0.0:
            record["AccuracyClu"] = utils.top1accuracy(scores_cluster, images_unlabeled_label)
        if patch_classification_loss_coef > 0.0:
            record["AccuracyPatch"] = utils.top1accuracy(scores_patch, labels_patches)

    if is_train:
        # Backward loss and apply gradient steps.
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        #att_classifier_optimizer.step()
        if rotation_loss_coef > 0.0:
            rot_classifier_optimizer.step()
        if patch_location_loss_coef > 0.0:
            location_classifier_optimizer.step()
            jig_classifier_optimizer.step()
        if cluster_loss_coef > 0.0:
            clu_classifier_optimizer.step()
        if patch_classification_loss_coef > 0.0:
            patch_classifier_optimizer.step()

    return record


def fewshot_classification_with_jigsaw_puzzle__selfsupervision(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    location_classifier,
    location_classifier_optimizer,
    patch_classifier,
    patch_classifier_optimizer,
    images_train,
    patches_train,
    labels_train,
    labels_train_1hot,
    images_test,
    patches_test,
    labels_test,
    is_train,
    base_ids=None,
    patch_location_loss_coef=1.0,
    patch_classification_loss_coef=1.0,
    combine="average",
    standardize_patches=True,
):
    """Forward-backward propagation routine for few-shot model extended
    with the auxiliary self-supervised task of predicting the relative location
    of patches."""
    pdb.set_trace()
    assert images_train.dim() == 5
    assert images_test.dim() == 5
    assert images_train.size(0) == images_test.size(0)
    assert images_train.size(2) == images_test.size(2)
    assert images_train.size(3) == images_test.size(3)
    assert images_train.size(4) == images_test.size(4)
    assert labels_train.dim() == 2
    assert labels_test.dim() == 2
    assert labels_train.size(0) == labels_test.size(0)
    assert labels_train.size(0) == images_train.size(0)

    assert patches_train.dim() == 6
    assert patches_train.size(0) == images_train.size(0)
    assert patches_train.size(1) == images_train.size(1)
    assert patches_train.size(2) == 9

    assert patches_test.dim() == 6
    assert patches_test.size(0) == images_test.size(0)
    assert patches_test.size(1) == images_test.size(1)
    assert patches_test.size(2) == 9

    meta_batch_size = images_train.size(0)
    num_train = images_train.size(1)
    num_test = images_test.size(1)

    if is_train:
        # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        if patch_location_loss_coef > 0.0:
            location_classifier_optimizer.zero_grad()
        if patch_classification_loss_coef > 0.0:
            patch_classifier_optimizer.zero_grad()

    record = {}
    with torch.no_grad():
        images_train = utils.convert_from_5d_to_4d(images_train)
        images_test = utils.convert_from_5d_to_4d(images_test)
        labels_test = labels_test.view(-1)
        images = torch.cat([images_train, images_test], dim=0)

        batch_size_train = images_train.size(0)
        batch_size_train_test = images.size(0)
        assert batch_size_train == meta_batch_size * num_train
        assert batch_size_train_test == meta_batch_size * (num_train + num_test)

        patches_train = utils.convert_from_6d_to_4d(patches_train)
        patches_test = utils.convert_from_6d_to_4d(patches_test)
        if standardize_patches:
            patches_train = utils.standardize_image(patches_train)
            patches_test = utils.standardize_image(patches_test)
        patches = torch.cat([patches_train, patches_test], dim=0)

        assert patches_train.size(0) == batch_size_train * 9
        assert patches.size(0) == batch_size_train_test * 9

    with torch.set_grad_enabled(is_train):
        # Extract features from the images.
        features = feature_extractor(images)
        # Extract features from the image patches.
        features_patches = feature_extractor(patches)

        # Perform object classification task.
        features_train = features[:batch_size_train]
        features_test = features[batch_size_train:batch_size_train_test]
        features_train = utils.add_dimension(features_train, meta_batch_size)
        features_test = utils.add_dimension(features_test, meta_batch_size)
        (
            classification_scores,
            loss_classsification,
        ) = fewshot_utils.few_shot_feature_classification(
            classifier,
            features_test,
            features_train,
            labels_train_1hot,
            labels_test,
            base_ids,
        )
        record["loss_cls"] = loss_classsification.item()
        loss_total = loss_classsification

        # Perform the self-supervised task of relative patch locatioon
        # prediction.
        if patch_location_loss_coef > 0.0:
            scores_location, loss_location, labels_loc = patch_location_task(
                location_classifier, features_patches
            )
            record["loss_loc"] = loss_location.item()
            loss_total = loss_total + loss_location * patch_location_loss_coef

        # Perform the auxiliary task of classifying patches.
        if patch_classification_loss_coef > 0.0:
            pdb.set_trace()
            features_patches = add_patch_dimension(features_patches)
            pdb.set_trace()
            assert features_patches.size(0) == batch_size_train_test
            assert features_patches.size(1) == 9
            features_patches = combine_multiple_patch_features(features_patches, combine)
            pdb.set_trace()

            features_patches_train = utils.add_dimension(
                features_patches[:batch_size_train], meta_batch_size
            )
            features_patches_test = utils.add_dimension(
                features_patches[batch_size_train:batch_size_train_test], meta_batch_size
            )

            scores_patch, loss_patch = fewshot_utils.few_shot_feature_classification(
                patch_classifier,
                features_patches_test,
                features_patches_train,
                labels_train_1hot,
                labels_test,
                base_ids,
            )
            record["loss_patch_cls"] = loss_patch.item()
            loss_total = loss_total + loss_patch * patch_classification_loss_coef

        # Because the total loss consists of multiple individual losses
        # (i.e., 3) scale it down by a factor of 0.5.
        loss_total = loss_total * 0.5

    with torch.no_grad():
        num_base = base_ids.size(1) if (base_ids is not None) else 0
        record = fewshot_utils.compute_accuracy_metrics(
            classification_scores, labels_test, num_base, record
        )
        if patch_location_loss_coef > 0.0:
            record["AccuracyLoc"] = utils.top1accuracy(scores_location, labels_loc)
        if patch_classification_loss_coef > 0.0:
            record = fewshot_utils.compute_accuracy_metrics(
                scores_patch, labels_test, num_base, record, string_id="Patch"
            )

    if is_train:
        # Backward loss and apply gradient steps.
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        if patch_location_loss_coef > 0.0:
            location_classifier_optimizer.step()
        if patch_classification_loss_coef > 0.0:
            patch_classifier_optimizer.step()

    return record
