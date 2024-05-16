import logging
import os
logger = logging.getLogger(__name__)


def save_stats(dataset, run_dir):
    is_training = dataset.is_training
    # don't drop anything
    dataset.is_training = False

    motion_stats_dir = os.path.join(run_dir, "motion_stats")
    os.makedirs(motion_stats_dir, exist_ok=True)

    text_stats_dir = os.path.join(run_dir, "text_stats")
    os.makedirs(text_stats_dir, exist_ok=True)

    from tqdm import tqdm
    import torch
    from src.normalizer import Normalizer

    logger.info("Compute motion embedding stats")
    motionfeats = torch.cat([x["x"] for x in tqdm(dataset)])
    mean_motionfeats = motionfeats.mean(0)
    std_motionfeats = motionfeats.std(0)

    motion_normalizer = Normalizer(base_dir=motion_stats_dir, disable=True)
    motion_normalizer.save(mean_motionfeats, std_motionfeats)

    logger.info("Compute text embedding stats")
    textfeats = torch.cat([x["tx"]["x"] for x in tqdm(dataset)])
    mean_textfeats = textfeats.mean(0)
    std_textfeats = textfeats.std(0)

    text_normalizer = Normalizer(base_dir=text_stats_dir, disable=True)
    text_normalizer.save(mean_textfeats, std_textfeats)

    # re enable droping
    dataset.is_training = is_training