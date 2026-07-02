"""Utilidad para inicializar experimentos de Comet ML con nomenclatura estructurada."""
import os
import comet_ml

WORKSPACE    = "kaggle-taa-freesound-audio-tagging"   # <- tu workspace de Comet
PROJECT_NAME = "trained_cnns"

def init_comet_experiment(config, run_version=1):
    """Crea un experimento en Comet, lo nombra, loguea parámetros y tags."""
    if config["fine_tuning"]:
        prefix = "FT"
    elif config["transfer_learning"]:
        prefix = "TL"
    else:
        prefix = "Baseline"

    experiment_name = f"{prefix}_{config['architecture_name']}_{config['train_dataset_type']}_v{run_version}"
    print(f"==> Inicializando experimento Comet: '{experiment_name}'")

    experiment = comet_ml.Experiment(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name=PROJECT_NAME,
        workspace=WORKSPACE,
        auto_metric_logging=False,   # <- las métricas las logueamos desde hist.history (sin duplicar)
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_histogram_activation_logging=True,
    )
    experiment.set_name(experiment_name)
    experiment.log_parameters(config)
    experiment.log_parameter("run_version", run_version)

    tags = [prefix.lower(), config["architecture_name"], config["train_dataset_type"]]
    if config.get("data_augmentation", False):
        tags.append("data-augmentation")
    if config.get("is_final_candidate", False):
        tags.append("final-candidate")
    experiment.add_tags(tags)
    return experiment