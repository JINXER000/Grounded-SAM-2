import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from groundedSamUtils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class ObjectRecPipeline:
    def __init__(self, grounding_model, text_prompt, img_path, sam2_checkpoint, sam2_model_config, device, output_dir, dump_json_results):
        # Hyperparameters and paths
        self.grounding_model_id = grounding_model
        self.text_prompt = text_prompt
        self.img_path = img_path
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_config = sam2_model_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.dump_json_results = dump_json_results

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Environment settings
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Build SAM2 image predictor
        self.sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # Build Grounding DINO from Huggingface
        self.processor = AutoProcessor.from_pretrained(self.grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_model_id).to(self.device)

    def preprocess_image_and_text(self, img_arr):
        self.sam2_predictor.set_image(img_arr)
        self.inputs = self.processor(img_arr, text=self.text_prompt, return_tensors="pt").to(self.device)

    def get_grounded_objects(self, img_arr):
        with torch.no_grad():
            self.outputs = self.grounding_model(**self.inputs)
        results = self.processor.post_process_grounded_object_detection(
            self.outputs,
            self.inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[img_arr.shape[::-1]]
        )
        return results[0]

    def predict_masks(self, input_boxes):
        input_boxes = input_boxes.cpu().numpy()
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False
        )
        return masks, scores

    def post_process_masks(self, masks):
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks

    def annotate_image(self, img, input_boxes, masks, class_names, confidences):
        class_ids = np.array(list(range(len(class_names))))
        labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences)]

        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        cv2.imwrite(os.path.join(self.output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

    def save_results(self, img_arr, img_path, class_names, input_boxes, masks, scores):
        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        if self.dump_json_results:
            mask_rles = [single_mask_to_rle(mask) for mask in masks]
            input_boxes = input_boxes.tolist()
            scores = scores.tolist()
            results = {
                "image_path": img_path,
                "annotations": [
                    {
                        "class_name": class_name,
                        "bbox": box,
                        "segmentation": mask_rle,
                        "score": score,
                    }
                    for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
                ],
                "box_format": "xyxy",
                "img_width": img_arr.shape[1],
                "img_height": img_arr.shape[0],
            }
            with open(os.path.join(self.output_dir, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
                json.dump(results, f, indent=4)

    def run(self):
        img = cv2.imread(self.img_path)
        # Preprocess the image and text prompt
        self.preprocess_image_and_text(img)

        # Get grounded objects from Grounding DINO
        grounded_objects = self.get_grounded_objects(img)
        input_boxes = grounded_objects["boxes"]
        class_names = grounded_objects["labels"]
        confidences = grounded_objects["scores"].cpu().numpy().tolist()

        # Predict masks with SAM2
        masks, scores = self.predict_masks(input_boxes)
        masks = self.post_process_masks(masks)

        # Annotate image with detections and masks
        self.annotate_image(img, input_boxes.cpu().numpy(), masks, class_names, confidences)

        # Save the results in JSON format
        self.save_results(img, self.img_path, class_names, input_boxes.cpu().numpy(), masks, scores)

if __name__ == "__main__":
    # Example usage
    model = ObjectRecPipeline(
        grounding_model="IDEA-Research/grounding-dino-tiny",
        text_prompt="car. tire.",
        img_path="notebooks/images/truck.jpg",
        sam2_checkpoint="./checkpoints/sam2_hiera_large.pt",
        sam2_model_config="sam2_hiera_l.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="outputs/grounded_sam2_hf_model_demo",
        dump_json_results=True
    )
    model.run()
