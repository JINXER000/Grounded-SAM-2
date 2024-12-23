import os
import cv2
import torch
import numpy as np
import supervision as sv

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from groundedSamUtils.track_utils import sample_points_from_masks
from groundedSamUtils.video_utils import create_video_from_images

class SAMTrackPipeline:
    def __init__(self, model_id = "IDEA-Research/grounding-dino-tiny", video_path = None, text_prompt=None, output_video_path = None, source_video_frame_dir = None, 
                 save_tracking_results_dir = None, prompt_type_for_video="box", sam2_checkpoint="./checkpoints/sam2_hiera_large.pt", 
                 model_cfg="sam2_hiera_l.yaml"):
        """
        Initializes the pipeline with required models, paths, and parameters.
        """
        
        self.model_id = model_id
        self.video_path = video_path
        self.text_prompt = text_prompt
        self.output_video_path = output_video_path
        self.source_video_frame_dir = source_video_frame_dir
        self.save_tracking_results_dir = save_tracking_results_dir
        self.prompt_type_for_video = prompt_type_for_video

        assert self.text_prompt is not None, "Text prompt is not provided"
        assert self.source_video_frame_dir is not None, "Source video frame directory is not provided"
        assert self.save_tracking_results_dir is not None, "Save tracking results directory is not provided"

        # Setup environment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load models
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        self.sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(self.sam2_image_model)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

    def deinit(self, inference_state):
        inference_state["images"].cpu()
        self.sam2_image_model.cpu()
        self.grounding_model.cpu()
        ## free cuda mem
        # del rgb_segmenter.video_predictor
        # del rgb_segmenter
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()


    def save_video_frames(self):
        """
        Converts video into frames and saves them to a specified directory.
        """
        assert self.video_path is not None, "Video path is not provided"
        
        video_info = sv.VideoInfo.from_video_path(self.video_path)
        print(video_info)
        frame_generator = sv.get_video_frames_generator(self.video_path, stride=1, start=0, end=None)
        
        source_frames = Path(self.source_video_frame_dir)
        source_frames.mkdir(parents=True, exist_ok=True)

        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)


    def save_images_frames(self, image_list):
        """
        Saves the provided list of images to the specified directory.
        
        Args:
        - image_list: A list of images (e.g., numpy arrays or PIL images).
        """
        source_frames = Path(self.source_video_frame_dir)
        source_frames.mkdir(parents=True, exist_ok=True)

        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for idx, frame in enumerate(image_list):
                sink.save_image(frame)

        print(f"Saved {len(image_list)} images to {self.source_video_frame_dir}")



    def get_bounding_boxes(self, frame_idx=0):
        """
        Prompts the grounding model to get bounding boxes for objects based on the text prompt.
        """
        img_path = os.path.join(self.source_video_frame_dir, self.frame_names[frame_idx])
        image = Image.open(img_path)
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        input_boxes = results[0]["boxes"].cpu().numpy()
        class_names = results[0]["labels"]

        self.image_predictor.set_image(np.array(image.convert("RGB")))

        print(input_boxes, class_names)
        return input_boxes, class_names

    def get_masks(self, input_boxes):
        """
        Prompts the SAM2 image predictor to generate masks based on bounding boxes.
        """
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks

    def add_points_or_box(self, inference_state, input_boxes, masks, objects, ann_frame_idx):
        """
        Registers the detected object points or boxes with the video predictor.
        """
        assert self.prompt_type_for_video in ["point", "box", "mask"], "Unsupported prompt type"

        if self.prompt_type_for_video == "point":
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
            for object_id, (label, points) in enumerate(zip(objects, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels
                )
        elif self.prompt_type_for_video == "box":
            for object_id, (label, box) in enumerate(zip(objects, input_boxes), start=1):
                self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box
                )
        elif self.prompt_type_for_video == "mask":
            for object_id, (label, mask) in enumerate(zip(objects, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                self.video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )

    def propagate_video(self, inference_state):
        """
        Propagates the segmentation results across the video frames.
        """
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments

    def save_results(self, video_segments, objects):
        """
        Saves the annotated segmentation results for each frame.
        """
        if not os.path.exists(self.save_tracking_results_dir):
            os.makedirs(self.save_tracking_results_dir)

        id_to_objects = {i: obj for i, obj in enumerate(objects, start=1)}

        for frame_idx, segments in video_segments.items():
            img = cv2.imread(os.path.join(self.source_video_frame_dir, self.frame_names[frame_idx]))
            object_ids = list(segments.keys())
            masks = np.concatenate(list(segments.values()), axis=0)


            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),
                mask=masks,
                class_id=np.array(object_ids, dtype=np.int32)
            )
            
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[id_to_objects[i] for i in object_ids])
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            cv2.imwrite(os.path.join(self.save_tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

        ## save the video
        if self.output_video_path is not None:
            create_video_from_images(self.save_tracking_results_dir, self.output_video_path)

    def run_pipeline(self):

        # After saving, scan all the JPEG frame names in this directory
        self.frame_names = sorted(
            [p for p in os.listdir(self.source_video_frame_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
            key=lambda p: int(os.path.splitext(p)[0])
        )        
        input_boxes, objects = self.get_bounding_boxes()
        masks = self.get_masks(input_boxes)
        inference_state = self.video_predictor.init_state(video_path=self.source_video_frame_dir)
        self.add_points_or_box(inference_state, input_boxes, masks, objects, ann_frame_idx=0)
        video_segments = self.propagate_video(inference_state)

        # self.deinit(inference_state)
        self.video_predictor.reset_state(inference_state)

        return video_segments, objects


if __name__ == "__main__":
    # MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    # TEXT_PROMPT = "hippopotamus."
    # OUTPUT_VIDEO_PATH = "./hippopotamus_tracking_demo.mp4"
    # SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
    # SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
    # PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
    # pipeline = SAMTrackPipeline(MODEL_ID, VIDEO_PATH, TEXT_PROMPT, OUTPUT_VIDEO_PATH, SOURCE_VIDEO_FRAME_DIR, SAVE_TRACKING_RESULTS_DIR)
    # pipeline.save_video_frames()

    # VIDEO_PATH = "./assets/screwdriver.mp4"
    VIDEO_PATH = None
    pipeline = SAMTrackPipeline(text_prompt = 'screwdriver.box.', video_path = VIDEO_PATH, source_video_frame_dir = "./custom_video_frames", \
                                          save_tracking_results_dir = "./tracking_results" )
    
    # pipeline.save_video_frames()
    
    video_segments, objects =  pipeline.run_pipeline()
    pipeline.save_results(video_segments, objects)



