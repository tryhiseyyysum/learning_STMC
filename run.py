from lib.config import cfg, args
import numpy as np
import os
import hydra
from omegaconf import DictConfig
from hydra import compose, initialize
from hydra.utils import instantiate
from lib.config.logger import save_stats
from src.config import read_config, save_config
import logging
import shutil
logger = logging.getLogger(__name__)

### SCRIPTS BEGINING ###

#@hydra.main(config_path="lib/datasets", config_name="train", version_base="1.3")
def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm


    #-----------------
    #initialize(config_path="lib/datasets", job_name="train", version_base="1.3")
    with initialize(config_path="lib/datasets", job_name="train", version_base="1.3"):
        cfg = compose(config_name="train.yaml")
    
    #cfg.train.num_workers = 0
    #----------------
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass

def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    #from lib.utils.data_utils import to_cuda
    import tqdm
    import torch
    import time

    #---------------------
    ckpt = None
    hydra.initialize(config_path="lib/datasets", job_name="train", version_base="1.3")
    cfg = compose(config_name="train.yaml")


    config_path = save_config(cfg)
    logger.info("Training script")
    logger.info(f"The config can be found here: \n{config_path}")
    import pytorch_lightning as pl

    pl.seed_everything(cfg.seed)

    logger.info("Loading the dataloaders")

    
    train_dataloader = make_data_loader(cfg, is_train=True)
    val_dataloader = make_data_loader(cfg, is_train=False)


    logger.info("Loading the model")
    diffusion = instantiate(cfg.diffusion)

    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(diffusion, train_dataloader, val_dataloader, ckpt_path=ckpt)

    


def run_generate():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    logger.info("Prediction script")

    hydra.initialize(config_path="lib/datasets", job_name="generate", version_base="1.3")
    c = compose(config_name="generate.yaml")

    assert c.input_type in ["text", "timeline", "auto"]
    assert c.baseline in ["none", "sinc", "sinc_lerp"]

    exp_folder_name = os.path.splitext(os.path.split(c.timeline)[-1])[0]

    if c.baseline != "none":
        exp_folder_name += "_baseline_" + c.baseline

    cfg = read_config(c.run_dir)
    fps = cfg.data.motion_loader.fps

    interval_overlap = int(fps * c.overlap_s)

    from src.stmc import read_timelines, process_timelines
    from src.text import read_texts

    if c.input_type == "auto" or "timeline":
        try:
            timelines = read_timelines(c.timeline, fps)
            logger.info("Reading the timelines")
            n_motions = len(timelines)
            c.input_type = "timeline"
        except IndexError:
            c.input_type = "text"
    if c.input_type == "text":
        logger.info("Reading the texts")
        texts_durations = read_texts(c.timeline, fps)
        n_motions = len(texts_durations)

    logger.info("Loading the libraries")
    import src.prepare  # noqa
    import pytorch_lightning as pl
    import numpy as np
    import torch

    if c.input_type == "text":
        infos = {
            "texts_durations": texts_durations,
            "all_lengths": [x.duration for x in texts_durations],
            "all_texts": [x.text for x in texts_durations],
        }
        infos["output_lengths"] = infos["all_lengths"]
    elif c.input_type == "timeline":
        infos = process_timelines(timelines, interval_overlap)
        infos["output_lengths"] = infos["max_t"]

        if c.baseline != "none":
            infos["baseline"] = c.baseline

    infos["featsname"] = cfg.motion_features
    infos["guidance_weight"] = c.guidance

    ckpt_name = c.ckpt
    ckpt_path = os.path.join(c.run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
    logger.info("Loading the checkpoint")

    ckpt = torch.load(ckpt_path, map_location=c.device)
    # Models
    logger.info("Loading the models")

    # Rendering
    joints_renderer = instantiate(c.joints_renderer)
    smpl_renderer = instantiate(c.smpl_renderer)

    # Diffusion model
    # update the folder first, in case it has been moved
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(c.run_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(c.run_dir, "text_stats")

    diffusion = instantiate(cfg.diffusion)
    diffusion.load_state_dict(ckpt["state_dict"])

    # Evaluation mode
    diffusion.eval()
    diffusion.to(c.device)

    # jointstype = "smpljoints"
    jointstype = "both"

    from src.tools.smpl_layer import SMPLH

    smplh = SMPLH(
        path="deps/smplh",
        jointstype=jointstype,
        input_pose_rep="axisangle",
        gender=c.gender,
    )

    from src.model.text_encoder import TextToEmb

    modelpath = cfg.data.text_encoder.modelname
    mean_pooling = cfg.data.text_encoder.mean_pooling
    text_model = TextToEmb(
        modelpath=modelpath, mean_pooling=mean_pooling, device=c.device
    )

    logger.info("Generate the function")

    video_dir = os.path.join(
        c.run_dir,
        "generations",
        exp_folder_name + "_" + str(ckpt_name) + f"_{c.input_type}_to_motion",
    )
    os.makedirs(video_dir, exist_ok=True)

    shutil.copy(
        c.timeline, os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}.txt")
    )

    vext = ".mp4"

    joints_video_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}_joints{vext}")
        for idx in range(n_motions)
    ]

    smpl_video_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}_smpl{vext}")
        for idx in range(n_motions)
    ]

    npy_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}.npy")
        for idx in range(n_motions)
    ]

    logger.info(f"All the output videos will be saved in: {video_dir}")

    if c.seed != -1:
        pl.seed_everything(c.seed)

    with torch.no_grad():
        tx_emb = text_model(infos["all_texts"])
        tx_emb_uncond = text_model(["" for _ in infos["all_texts"]])

        if isinstance(tx_emb, torch.Tensor):
            tx_emb = {
                "x": tx_emb[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(c.device),
            }
            tx_emb_uncond = {
                "x": tx_emb_uncond[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(
                    c.device
                ),
            }

        xstarts = diffusion(tx_emb, tx_emb_uncond, infos).cpu()

        for idx, (xstart, length) in enumerate(zip(xstarts, infos["output_lengths"])):
            xstart = xstart[:length]

            from src.tools.extract_joints import extract_joints

            output = extract_joints(
                xstart,
                infos["featsname"],
                fps=fps,
                value_from=c.value_from,
                smpl_layer=smplh,
            )

            joints = output["joints"]
            path = npy_paths[idx]
            np.save(path, joints)

            if "vertices" in output:
                path = npy_paths[idx].replace(".npy", "_verts.npy")
                np.save(path, output["vertices"])

            if "smpldata" in output:
                path = npy_paths[idx].replace(".npy", "_smpl.npz")
                np.savez(path, **output["smpldata"])

            logger.info(f"Joints rendering {idx}")
            joints_renderer(
                joints, title="", output=joints_video_paths[idx], canonicalize=False
            )
            print(joints_video_paths[idx])
            print()

            if "vertices" in output and not c.fast:
                logger.info(f"SMPL rendering {idx}")
                smpl_renderer(
                    output["vertices"], title="", output=smpl_video_paths[idx]
                )
                print(smpl_video_paths[idx])
                print()

            logger.info("Rendering done")


def run_generate_mtt():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    logger = logging.getLogger(__name__)

    logger.info("Prediction script")

    hydra.initialize(config_path="lib/datasets", job_name="generate", version_base="1.3")
    c = compose(config_name="generate.yaml")

    assert c.baseline in ["none", "sinc", "sinc_lerp", "singletrack", "onetext"]

    mtt_name = "mtt"
    if c.baseline == "onetext":
        mtt_file = "datasets/mtt/baselines/MTT_onetext.txt"
    elif c.baseline == "singletrack":
        mtt_file = "datasets/mtt/baselines/MTT_singletrack.txt"
    else:
        mtt_file = "datasets/mtt/MTT.txt"

    cfg = read_config(c.run_dir)
    fps = cfg.data.motion_loader.fps

    interval_overlap = int(fps * c.overlap_s)

    from src.stmc import read_timelines, process_timelines

    logger.info("Reading the timelines")
    all_timelines = read_timelines(mtt_file, fps)

    n_sequences = len(all_timelines)

    logger.info("Loading the libraries")
    import src.prepare  # noqa
    import pytorch_lightning as pl
    import numpy as np
    import torch
    from src.model.text_encoder import TextToEmb
    from src.tools.extract_joints import extract_joints

    ckpt_name = c.ckpt
    ckpt_path = os.path.join(c.run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
    logger.info("Loading the checkpoint")

    ckpt = torch.load(ckpt_path, map_location=c.device)
    # Models
    logger.info("Loading the models")

    # Diffusion model
    # update the folder first, in case it has been moved
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(c.run_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(c.run_dir, "text_stats")

    diffusion = instantiate(cfg.diffusion)
    diffusion.load_state_dict(ckpt["state_dict"])

    # Evaluation mode
    diffusion.eval()
    diffusion.to(c.device)

    # in case we want to get the joints from SMPL
    if c.value_from == "smpl":
        jointstype = "both"
        from src.tools.smpl_layer import SMPLH

        smplh = SMPLH(
            path="deps/smplh",
            jointstype=jointstype,
            input_pose_rep="axisangle",
            gender="neutral",
        )
    else:
        smplh = None

    modelpath = cfg.data.text_encoder.modelname
    mean_pooling = cfg.data.text_encoder.mean_pooling
    text_model = TextToEmb(
        modelpath=modelpath, mean_pooling=mean_pooling, device=c.device
    )

    out_path = os.path.join(
        c.run_dir,
        f"{mtt_name}_generations_" + str(ckpt_name) + "_from_" + c.value_from,
    )

    if c.baseline != "none":
        out_path += "_baseline_" + c.baseline

    if c.overlap_s != 0.5:
        out_path += "_intervaloverlap_" + str(c.overlap_s)

    os.makedirs(out_path, exist_ok=True)
    logger.info(f"The results (joints) will be saved in: {out_path}")

    if c.seed != -1:
        pl.seed_everything(c.seed)

    at_a_time = 50
    iterator = np.array_split(np.arange(n_sequences), n_sequences // at_a_time)

    with torch.no_grad():
        for x in iterator:
            timelines = [all_timelines[y] for y in x]
            npy_paths = [os.path.join(out_path, str(y).zfill(4) + ".npy") for y in x]

            if "sinc" in c.baseline:
                # No extension and no unconditional transitions
                infos = process_timelines(
                    timelines, interval_overlap, extend=False, uncond=False
                )
            else:
                infos = process_timelines(timelines, interval_overlap)

            infos["baseline"] = c.baseline
            infos["output_lengths"] = infos["max_t"]
            infos["featsname"] = cfg.motion_features
            infos["guidance_weight"] = c.guidance

            tx_emb = text_model(infos["all_texts"])
            tx_emb_uncond = text_model(["" for _ in infos["all_texts"]])

            if isinstance(tx_emb, torch.Tensor):
                tx_emb = {
                    "x": tx_emb[:, None],
                    "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(
                        c.device
                    ),
                }
                tx_emb_uncond = {
                    "x": tx_emb_uncond[:, None],
                    "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(
                        c.device
                    ),
                }

            xstarts = diffusion(tx_emb, tx_emb_uncond, infos).cpu()

            for idx, (length, npy_path) in enumerate(zip(infos["max_t"], npy_paths)):
                xstart = xstarts[idx, :length]
                output = extract_joints(
                    xstart,
                    infos["featsname"],
                    fps=fps,
                    value_from=c.value_from,
                    smpl_layer=smplh,
                )
                joints = output["joints"]

                # shape T, F
                np.save(npy_path, joints)


def run_evaluate():
    from lib.evaluators.stmc import read_timelines

    from lib.evaluators.metrics import calculate_activation_statistics_normalized
    from lib.evaluators.load_tmr_model import load_tmr_model_easy
    from lib.evaluators.load_mtt_texts_motions import load_mtt_texts_motions
    from lib.evaluators.stmc_metrics import get_gt_metrics, get_exp_metrics, print_latex

    from lib.evaluators.experiments import experiments

    device = "cpu"

    # FOLDER to evaluate
    fps = 20.0
    mtt_timelines = "../datasets/mtt/MTT.txt"

    tmr_forward = load_tmr_model_easy(device)
    texts_gt, motions_guofeats_gt = load_mtt_texts_motions(fps)

    text_dico = {t: i for i, t in enumerate(texts_gt)}

    text_latents_gt = tmr_forward(texts_gt)
    motion_latents_gt = tmr_forward(motions_guofeats_gt)

    metric_names = [32 * " ", "R1  ", "R3  ", "M2T S", "M2M S", "FID+ ", "Trans"]
    print(" & ".join(metric_names))

    gt_metrics = get_gt_metrics(motion_latents_gt, text_latents_gt, motions_guofeats_gt)
    print_latex("GT", gt_metrics)

    gt_mu, gt_cov = calculate_activation_statistics_normalized(motion_latents_gt.numpy())

    timelines = read_timelines(mtt_timelines, fps)
    assert len(timelines) == 500
    timelines_dict = {str(idx).zfill(4): timeline for idx, timeline in enumerate(timelines)}

    for exp in experiments:
        if exp.get("skip", False):
            continue
        metrics = get_exp_metrics(
            exp,
            tmr_forward,
            text_dico,
            timelines_dict,
            gt_mu,
            gt_cov,
            text_latents_gt,
            motion_latents_gt,
            fps,
        )
        print_latex(exp["name"], metrics)


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.utils.data_utils import to_cuda

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = network(batch)
        visualizer.visualize(output, batch)
    if visualizer.write_video:
        visualizer.summarize()

if __name__ == '__main__':
    globals()['run_' + args.type]()
