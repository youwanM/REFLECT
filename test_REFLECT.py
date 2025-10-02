
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import yaml
from scipy.ndimage import gaussian_filter
from medical_models import UNET_models
from MedicalDataLoader import BraTS2021Dataset, ATLASDataset
from huggingface_hub import hf_hub_download
from anomalib import metrics
from sklearn.metrics import average_precision_score
import cv2
from PIL import Image
from skimage.transform import resize
import copy
import pandas as pd
from scipy.ndimage import label

def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask


def compute_dice(anomaly_map, segmentation, th):
    anomaly_map[anomaly_map>th]=1
    anomaly_map[anomaly_map<1]=0
    if sum(segmentation.flatten())==0:
        if sum(anomaly_map.flatten())==0:
            return 1
        else:
            return 0
    
    eps = 1e-6
    # flatten label and prediction tensors
    inputs = anomaly_map.flatten()
    targets = segmentation.flatten()

    intersection = (inputs * targets).sum()
    dice = (2. * intersection) / (inputs.sum() + targets.sum() + eps)
    return dice

def compute_f1_per_lesion_pixel(anomaly_map, segmentation, threshold):
    """
    Compute lesion-level F1 score.

    A predicted lesion counts as TP if any pixel overlaps a GT lesion.
    Otherwise, it counts as FP. GT lesions with no overlapping prediction are FN.

    Args:
        anomaly_map (np.ndarray): predicted probability map
        segmentation (np.ndarray): binary ground truth mask
        threshold (float): threshold to binarize the prediction

    Returns:
        f1_score, TP, FP, FN
    """
    # Binarize predicted map
    pred_bin = (anomaly_map >= threshold).astype(np.uint8)

    # Label connected components
    labeled_gt, num_gt = label(segmentation)
    labeled_pred, num_pred = label(pred_bin)

    TP = 0
    gt_detected = np.zeros(num_gt, dtype=bool)

    for i in range(1, num_pred + 1):
        pred_lesion = (labeled_pred == i)

        # find all GT lesions that overlap with this predicted lesion
        overlap = labeled_gt[pred_lesion]
        overlap_labels = np.unique(overlap)
        overlap_labels = overlap_labels[overlap_labels > 0]  # remove background

        lesion_matched = False
        for label_id in overlap_labels:
            gt_detected[label_id - 1] = True
            lesion_matched = True  # TP if it touches at least one pixel

        if lesion_matched:
            TP += 1

    FP = num_pred - TP
    FN = num_gt - np.sum(gt_detected)

    f1_score = 2 * TP / (2 * TP + FP + FN) if (2*TP + FP + FN) > 0 else 0
    return f1_score


def compute_f1_per_lesion_min_overlap(anomaly_map, segmentation, threshold, min_overlap):
    """
    Compute lesion-level F1 score with minimal overlap requirement.

    A predicted lesion counts as TP if it overlaps any GT lesion by at least `min_overlap` fraction of the GT lesion.
    Otherwise, it counts as FP. GT lesions with no overlapping prediction are FN.

    Args:
        anomaly_map (np.ndarray): predicted probability map
        segmentation (np.ndarray): binary ground truth mask
        threshold (float): threshold to binarize the prediction
        min_overlap (float): minimal fraction of GT lesion pixels that must overlap with prediction

    Returns:
        f1_score, TP, FP, FN
    """
    # Binarize predicted map
    pred_bin = (anomaly_map >= threshold).astype(np.uint8)

    # Label connected components
    labeled_gt, num_gt = label(segmentation)
    labeled_pred, num_pred = label(pred_bin)

    TP = 0
    gt_detected = np.zeros(num_gt, dtype=bool)

    for i in range(1, num_pred + 1):
        pred_lesion = (labeled_pred == i)

        # find all GT lesions that overlap with this predicted lesion
        overlap = labeled_gt[pred_lesion]
        overlap_labels = np.unique(overlap)
        overlap_labels = overlap_labels[overlap_labels > 0]  # remove background

        lesion_matched = False
        for label_id in overlap_labels:
            gt_lesion = (labeled_gt == label_id)
            intersection = np.sum(pred_lesion & gt_lesion)
            gt_size = np.sum(gt_lesion)

            if intersection / gt_size >= min_overlap:
                lesion_matched = True
                gt_detected[label_id - 1] = True

        if lesion_matched:
            TP += 1

    FP = num_pred - TP
    FN = num_gt - np.sum(gt_detected)

    f1_score = 2 * TP / (2 * TP + FP + FN) if (2*TP + FP + FN) > 0 else 0
    return f1_score
def per_image_f1_components_df(anomaly_maps, anoGT, segmentation, filenames, threshold):
    """
    Compute component-level F1 score per image using a fixed threshold
    and return as a pandas DataFrame.
    """
    sub_to_idx = {fname: i for i, fname in enumerate(filenames)}
    structure = np.ones((3,3), dtype=int)  # 8-connectivity

    f1_list = []

    subjects = anoGT['SubjectID'].unique()

    for sub in subjects:
        if sub not in sub_to_idx:
            print(f"Warning: SubjectID {sub} not in filenames.")
            continue

        idx = sub_to_idx[sub]
        pred_mask = (anomaly_maps[idx] >= threshold).astype(np.uint8)
        seg_mask = segmentation[idx].astype(bool)
        
        # connected components in prediction
        labeled_pred, num_pred = label(pred_mask, structure=structure)
        pred_matched = np.zeros(num_pred, dtype=bool)

        # GT coordinates for this subject
        coords = anoGT[anoGT['SubjectID'] == sub][['x', 'y']].values
        if not 2 in (anoGT[anoGT['SubjectID'] == sub][['class_id']].values).flatten():
            continue
        gt_matched = np.zeros(len(coords), dtype=bool)

        for gi, (x, y) in enumerate(coords):
            y_int, x_int = int(round(y)), int(round(x))
            # skip GT points inside segmentation mask
            if 0 <= y_int < seg_mask.shape[0] and 0 <= x_int < seg_mask.shape[1]:
                if seg_mask[y_int, x_int]:
                    continue  # ignore this GT point
                comp_id = labeled_pred[y_int, x_int]
                if comp_id > 0:
                    pred_matched[comp_id - 1] = True
                    gt_matched[gi] = True

        TP = np.sum(pred_matched)
        FP = num_pred - TP
        FN = np.sum(~gt_matched)

        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        f1_list.append({'SubjectID': sub, 'F1': f1})

    return pd.DataFrame(f1_list)


def optimize_f1_components(anomaly_maps, anoGT, segmentations, filenames, thresholds=np.arange(0.01, 1.0, 0.01)):
    """
    Find the threshold that maximizes the average component-level F1 score across images.
    """
    best_avg_f1 = -1
    best_threshold = None

    for t in thresholds:
        f1_df = per_image_f1_components_df(anomaly_maps, anoGT, segmentations, filenames, threshold=t)
        avg_f1 = f1_df['F1'].mean() if not f1_df.empty else 0

        if avg_f1 >= best_avg_f1:
            best_avg_f1 = avg_f1
            best_threshold = t

    return best_threshold, best_avg_f1

def compute_metrics_per_subject(anomaly_maps, segmentations, filenames, threshold):
    """
    Compute pixel-wise Dice and lesion-level F1 per subject,
    stratify by lesion size, and return a DataFrame.
    """
    dice_scores = []
    f1_scores_overlap = []
    f1_scores_1pixel = []
    lesion_volumes = []

    min_overlap = 0.1

    for k in range(len(anomaly_maps)):
        anomaly_map = np.asarray(anomaly_maps[k])
        segmentation = np.asarray(segmentations[k])

        # Dice
        d = compute_dice(anomaly_map, segmentation, threshold)
        dice_scores.append(d)

        # F1 per lesion (overlap)
        f1_overlap = compute_f1_per_lesion_min_overlap(anomaly_map, segmentation, threshold, min_overlap)
        f1_scores_overlap.append(f1_overlap)

        # F1 per lesion (1pixel)
        f1_1pix = compute_f1_per_lesion_pixel(anomaly_map, segmentation, threshold)
        f1_scores_1pixel.append(f1_1pix)

        # Lesion volume
        lesion_volumes.append(np.sum(segmentation))

    # Build dataframe
    df = pd.DataFrame({
        "SubjectID": filenames,
        "Dice": dice_scores,
        "F1_Overlap": f1_scores_overlap,
        "F1_1pixel": f1_scores_1pixel,
        "LesionVolume": lesion_volumes
    })

    # Drop subjects with no lesions
    df = df[df["LesionVolume"] > 0].reset_index(drop=True)

    # Compute size thresholds
    small_thresh = np.percentile(df["LesionVolume"], 25)
    large_thresh = np.percentile(df["LesionVolume"], 75)

    def assign_category(v):
        if v <= small_thresh:
            return "S"
        elif v >= large_thresh:
            return "L"
        else:
            return "M"

    df["SizeCategory"] = df["LesionVolume"].apply(assign_category)
    #print(df)
    #print('\n')
    print(f"All sizes: Mean Dice = {df['Dice'].mean():.4f}, "
        f"Mean F1 10% overlap = {df['F1_Overlap'].mean():.4f}, "
        f"Mean F1 1px overlap = {df['F1_1pixel'].mean():.4f}")

    # Print mean metrics per size
    for size in ["S", "M", "L"]:
        mean_dice = df.loc[df["SizeCategory"] == size, "Dice"].mean()
        mean_f1_overlap = df.loc[df["SizeCategory"] == size, "F1_Overlap"].mean()
        mean_f1_1pixel = df.loc[df["SizeCategory"] == size, "F1_1pixel"].mean()
        print(f"Size {size}: Mean Dice = {mean_dice:.4f}, Mean F1 {100*min_overlap:.1f}% overlap= {mean_f1_overlap:.4f}, Mean F1 1px overlap= {mean_f1_1pixel:.4f}")

    return df


    
def calculate_metrics(ground_truth, prediction, filenames, threshold):
    flat_gt = ground_truth.flatten()
    flat_pred = prediction.flatten()
    
    #Anomaly Detection score
    anoGT = pd.read_csv(os.path.join(args.data_dir, "anoGT.csv"))
    anoThr, anoF1 = optimize_f1_components(prediction, anoGT, ground_truth, filenames)
    per_image_f1_components_df(prediction, anoGT, ground_truth, filenames, threshold=anoThr).to_csv(os.path.join(args.parent_dir, f'lesion_f1_per_image_{args.backward_steps}_backward_steps_test.csv'), index=False)

    metrics_df  = compute_metrics_per_subject(prediction, ground_truth, filenames, threshold)


    print(f"Anomaly detection optimal threshold: {anoThr:.4f}, F1: {anoF1:.4f}")
    
    auroc = metrics.AUROC()
    auroc_score = auroc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    f1max = metrics.F1Max()
    f1_max_score = f1max(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))
    
    ap = average_precision_score(ground_truth.flatten(), prediction.flatten())
    
    return auroc_score.cpu().numpy() ,f1_max_score.cpu().numpy(), ap, metrics_df


def visualize(anomaly_maps, segmentations, xs, image_samples, filenames, args):
    counter = -1
    base_dir = os.path.join(args.parent_dir, f'visualization/test_{args.backward_steps}_backward_steps/')
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'inputs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'overlays'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'counterfactuals'), exist_ok=True)

    for anomaly_map, segmentation, x, image_sample, filename in zip(anomaly_maps, segmentations, xs, image_samples, filenames):
        counter += 1
        visualization_image = np.zeros((4*args.image_size, args.image_size, 3), dtype=np.uint8)

        # --- Input image ---
        input_image = ((np.clip(x[0].detach().cpu().numpy(), -1, 1).transpose(1, 2, 0))
                       * 127.5 + 127.5).astype(np.uint8)
        if input_image.shape[-1] == 1:  # (H, W, 1) → (H, W, 3)
            input_image = np.repeat(input_image, 3, axis=-1)

        # --- Reconstructed image ---
        output_image = ((np.clip(image_sample[0].detach().cpu().numpy(), -1, 1).transpose(1, 2, 0))
                        * 127.5 + 127.5).astype(np.uint8)
        if output_image.shape[-1] == 1:  # (H, W, 1) → (H, W, 3)
            output_image = np.repeat(output_image, 3, axis=-1)

        # --- Anomaly map overlay ---
        scoremap = cv2.applyColorMap((anomaly_map*255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        anomal_map_img = (0.5 * input_image + 0.5 * scoremap).astype(np.uint8)

        # --- Segmentation mask ---
        seg_arr = segmentation.detach().cpu().numpy()
        if seg_arr.ndim == 3:   # (1, H, W)
            seg_arr = seg_arr[0]
        seg_arr = (seg_arr > 0).astype(np.uint8) * 255
        seg_vis = np.stack([seg_arr]*3, axis=-1)  # (H, W, 3)

        # --- Stacked visualization ---
        visualization_image[:args.image_size, :] = input_image
        visualization_image[args.image_size:2*args.image_size, :] = output_image
        visualization_image[2*args.image_size:3*args.image_size, :] = seg_vis
        visualization_image[3*args.image_size:, :] = anomal_map_img

        # --- Save everything ---
        Image.fromarray(visualization_image).save(os.path.join(base_dir, f'{filename}_visualization.png'))
        Image.fromarray(input_image).save(os.path.join(base_dir, f'inputs/{filename}_input.png'))
        Image.fromarray(output_image).save(os.path.join(base_dir, f'counterfactuals/{filename}_counterfactual.png'))

        # anomaly_map is 2D (H, W)
        anomaly_gray = (anomaly_map*255).astype(np.uint8)
        if anomaly_gray.ndim == 3 and anomaly_gray.shape[0] == 1:  # squeeze (1,H,W)
            anomaly_gray = anomaly_gray[0]
        #Image.fromarray(anomaly_gray, mode="L").save(os.path.join(base_dir, f'{filename}_anomaly.png'))

        Image.fromarray(anomal_map_img).save(os.path.join(base_dir, f'overlays/{filename}_overlay.png'))
        #Image.fromarray(seg_vis).save(os.path.join(base_dir, f'{counter}_mask.png'))


def evaluate(x0s, segmentations, encodeds,  image_samples, latent_samples, filenames, args):
        anomaly_maps = []
        gt = []

        for x, segmentation, encoded, image_sample, latent_sample in zip(x0s, segmentations, encodeds,  image_samples, latent_samples):
                image_difference = (((((torch.abs(image_sample-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
                image_difference = (np.clip(image_difference, 0.0, 0.4) ) * 2.5

                image_difference = smooth_mask(image_difference, sigma=3)
                
                latent_difference = (((((torch.abs(latent_sample-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
                latent_difference = (np.clip(latent_difference, 0.0 , 0.4)) * 2.5
                latent_difference = smooth_mask(latent_difference, sigma=1)
                latent_difference = resize(latent_difference, (args.image_size, args.image_size))
                
                final_anomaly = 1/2*image_difference + 1/2*latent_difference
                anomaly_maps.append(final_anomaly)
                gt.append(segmentation[0,:,:].cpu().numpy())
                
        anomaly_maps = np.stack(anomaly_maps, axis=0)
        
        gt = np.stack(gt, axis=0)
        gt = (gt>0).astype(np.int32)

        auroc_score ,f1_max_score, ap, metrics_df = calculate_metrics(gt, anomaly_maps, filenames, args.threshold)
        save_path = os.path.join(args.parent_dir, f"dice_stratification_{args.threshold}_test.csv")
        metrics_df.to_csv(save_path, index=False)
        with open(os.path.join(args.parent_dir, f'results_with_{args.backward_steps}_backward_steps_test.txt'), 'w') as f:
            f.write('Threshold:{:.4f}\nGlobal max Dice score: {:.4f}\nAUROC: {:.4f}\nAP: {:.4f}'.format(
                np.round(args.threshold, 4),
                np.round(f1_max_score, 4),
                np.round(auroc_score, 4),
                np.round(ap,4)
            ))
            
        return anomaly_maps, {'max Dice score':np.round(metrics_df['Dice'].mean(), 4),'max Dice score threshold':np.round(args.threshold, 4), 'Global max Dice score': np.round(f1_max_score, 4), 'AUROC':np.round(auroc_score, 4), 'AP':np.round(ap,4)}




def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    if args.vae == 'kl_f4':
        in_channels = out_channels = 3
    else:
        in_channels = out_channels = 4
        
    model = UNET_models[args.model](in_channels=in_channels, out_channels=out_channels)
    try:
        state_dict = torch.load(args.model_path)['model']
        print(model.load_state_dict(state_dict))
    except:
        raise Exception('Provided trained model path could not be found or it is not consistent with model params.')
    
    model.eval()
    model.to(device)    

    if args.vae == 'kl_f4':
        vae_model_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename="VAE-Medical-klf4.pt")
        vae = torch.load(vae_model_path)
        embedding_dim = 3
        compression_factor = 4
        
    elif args.vae == 'kl_f8':
        vae_model_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename="VAE-Medical-klf8.pt")
        vae = torch.load(vae_model_path)
        embedding_dim = 4
        compression_factor = 8
        
    vae.eval()
    vae.to(device)
    
    # Setup data:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])
    
    if args.dataset == 'BraTS':
        test_dataset = BraTS2021Dataset('test', rootdir=args.data_dir, transform=transform, image_size=args.image_size, augment=False, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    else:
        test_dataset = ATLASDataset('test', rootdir=args.data_dir, transform=transform, image_size=args.image_size, augment=False, modality=args.modality, embedding_dim=embedding_dim, compression_factor=compression_factor)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    
    print(f"Dataset contains {len(test_dataset)} Test images.")

    x_s = []
    encoded_s = []
    image_samples_s = []
    latent_samples_s = []
    x0_s = []
    segmentation_s = []
    filenames = []
    
    print('=-='*20)
    print('Starting evaluation...')
    print('=-='*20)
    for ii, (x, mask, seg, fname) in enumerate(test_loader):
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            encoded = vae.encode(x.to(device)).mean.mul_(0.18215)#Normalization params got from LDM package
            
            latent_sample = encoded.clone()
        # Euler solver (can use higher-order methods)
            dt = 1 / args.backward_steps   # Step size (adjust for accuracy/speed tradeoff)
            for time in torch.arange(0, 1, dt):
                t = time * torch.ones((encoded.shape[0], 1)).to(torch.float32).to(device)
                # velocity = model(encoded, t)
                velocity = model(latent_sample, t)
                latent_sample = latent_sample + velocity * dt

            image_samples = vae.decode(latent_sample / 0.18215) #* (1-mask)
            x0 = vae.decode(encoded / 0.18215) #* (1-mask)

            x_s += [_x.unsqueeze(0) for _x in x]
            segmentation_s += [_seg.unsqueeze(0) for _seg in seg]
            encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
            image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_sample]
            x0_s += [_x0.unsqueeze(0) for _x0 in x0]
            filenames += [os.path.splitext(os.path.basename(f))[0] for f in fname]


    anomaly_maps, results = evaluate(x0_s, segmentation_s, encoded_s,  image_samples_s, latent_samples_s, filenames, args)
    for key, val in results.items():
        print(key, ' : ', val)
    visualize(anomaly_maps, segmentation_s, x0_s, image_samples_s, filenames, args)
    print('=-='*20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default='Data/')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--backward-steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    

    args = parser.parse_args()
    
    args.parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.model_path)))
    try:
        with open(os.path.join(args.parent_dir, 'args.yml'), 'r') as file:
            config = yaml.safe_load(file)  
        args.dataset = config['dataset']
        args.image_size = int(config['image_size'])
        args.modality = config['modality']
        args.vae = config['vae']
        args.model = config['model']          
    except:
        raise Exception("YAML config file could not be found in the parent folder of the provided model path")

    main(args)
    



