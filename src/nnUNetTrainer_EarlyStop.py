# my_trainer_earlystop.py
import os, json
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.evaluation.evaluate_predictions import (
    compute_metrics_on_folder,
    labels_to_list_of_regions,
)
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


class nnUNetTrainer_EarlyStop(nnUNetTrainer):
    def __init__(self, *args, patience=5, min_delta=0.002, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self._best = 0.0
        self._bad = 0
        self._stop_now = False

    def validate(self, *args, **kwargs):
        # run the built-in validation (writes predictions into output_folder/validation)
        ret = super().validate(*args, **kwargs)

        # compute mean Dice across classes on those predictions
        pred = os.path.join(self.output_folder, "validation")
        gt = (
            os.path.join(self.nnunet_raw, self.dataset_name, "labelsTr")
            if hasattr(self, "nnunet_raw")
            else None
        )
        dj = (
            os.path.join(self.nnunet_raw, self.dataset_name, "dataset.json")
            if hasattr(self, "nnunet_raw")
            else None
        )
        outj = os.path.join(pred, f"summary_epoch{self.current_epoch}.json")

        # Fallback: if you don’t have absolute roots exposed, pass absolute paths via env or store them in self at init.
        rw = SimpleITKIO()
        # derive label list {1,2,3,...} from label_manager
        labels = sorted([k for k in self.label_manager.all_labels.keys() if k != 0])
        regions = labels_to_list_of_regions(labels)
        file_ending = self.label_manager.file_ending

        compute_metrics_on_folder(
            gt, pred, outj, rw, file_ending, regions, ignore_label=None, num_processes=4
        )

        # parse mean Dice
        with open(outj) as f:
            d = json.load(f)
        items = d.get("results") or d.get("cases") or (d if isinstance(d, list) else [])
        dices = []
        for it in items:
            m = it.get("metrics", {})
            for _, clsd in m.items():
                if isinstance(clsd, dict) and "Dice" in clsd:
                    dices.append(float(clsd["Dice"]))
        mean_dice = sum(dices) / len(dices) if dices else 0.0
        self.print_to_log_file(
            f"[earlystop] mean Dice={mean_dice:.4f} (best={self._best:.4f}, bad={self._bad}/{self.patience})"
        )

        if mean_dice > self._best + self.min_delta:
            self._best = mean_dice
            self._bad = 0
        else:
            self._bad += 1
            if self._bad >= self.patience:
                self._stop_now = True
                self.print_to_log_file("[earlystop] patience reached → requesting stop")

        return ret

    def run_training(self):
        super().run_training()
        # If your nnU-Net version doesn't check a stop flag inside the loop,
        # you can instead lower max_num_epochs at the point _stop_now is set,
        # or raise SystemExit to exit cleanly when _stop_now is True.
