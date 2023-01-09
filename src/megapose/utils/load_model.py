# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObjectDataset
from megapose.inference.icp_refiner import ICPRefiner
from megapose.inference.pose_estimator import PoseEstimator
from megapose.inference.utils import load_pose_models

NAMED_MODELS = {
    "megapose-1.0-RGB": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": False,
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 1,
        },
    },
    "megapose-1.0-RGBD": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgbd-288182519",
        "requires_depth": True,
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 1,
        },
    },
    "megapose-1.0-RGB-multi-hypothesis": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": False,
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 5,
        },
    },
    "megapose-1.0-RGB-multi-hypothesis-icp": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": True,
        "depth_refiner": "ICP",
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 5,
            "run_depth_refiner": True,
        },
    },
}


def load_named_model(
    model_name: str,
    object_dataset: RigidObjectDataset,
    n_workers: int = 4,
    bsz_images: int = 128,
) -> PoseEstimator:

    model = NAMED_MODELS[model_name]

    renderer_kwargs = {
        "preload_cache": False,
        "split_objects": False,
        "n_workers": n_workers,
    }

    coarse_model, refiner_model, mesh_db = load_pose_models(
        coarse_run_id=model["coarse_run_id"],
        refiner_run_id=model["refiner_run_id"],
        object_dataset=object_dataset,
        force_panda3d_renderer=True,
        renderer_kwargs=renderer_kwargs,
        models_root=LOCAL_DATA_DIR / "megapose-models",
    )

    depth_refiner = None
    if model.get("depth_refiner", None) == "ICP":
        depth_refiner = ICPRefiner(
            mesh_db,
            refiner_model.renderer,
        )

    pose_estimator = PoseEstimator(
        refiner_model=refiner_model,
        coarse_model=coarse_model,
        detector_model=None,
        depth_refiner=depth_refiner,
        bsz_objects=8,
        bsz_images=bsz_images,
    )
    return pose_estimator
