import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_imshow,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=["yolov5s.pt"],
        help="Path(s) to model .pt file(s)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images",
        help="File/folder, URL, or webcam (e.g. 0)."
    )
    parser.add_argument(
        "--results-loc",
        type=str,
        default="runs/detect",
        help="Where to store the results.txt file (in/out counts)."
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference size (pixels)."
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Object confidence threshold."
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="IOU threshold for NMS."
    )
    parser.add_argument(
        "--device",
        default="",
        help="CUDA device (e.g. 0 or 0,1,2,3) or 'cpu'."
    )
    parser.add_argument(
        "--view-img",
        action="store_true",
        help="Display results in a window."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="Filter by class: --classes 0 or --classes 0 2 3."
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Class-agnostic NMS."
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augmented inference."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update all models (to fix SourceChangeWarning)."
    )
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Save results to project/name."
    )
    parser.add_argument(
        "--name",
        default="exp",
        help="Save results to project/name."
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Existing project/name is okay, do not increment."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save inference results (image or video)."
    )
    return parser.parse_args()


def detect(args: argparse.Namespace) -> None:
    """
    Runs object detection using YOLOv5 and Deep SORT for tracking. 
    Also counts 'in' and 'out' crosses of a horizontal line in the frame.

    :param args: Command-line arguments from parse_args().
    """
    # Parameters for Deep SORT
    max_cosine_distance: float = 0.4
    nn_budget: Optional[int] = None
    nms_max_overlap: float = 1.0

    # Initialize Deep SORT
    model_filename: str = "weights/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        metric="cosine",
        matching_threshold=max_cosine_distance,
        budget=nn_budget
    )
    tracker = Tracker(metric, max_age=60, max_iou_distance=0.7, n_init=3)

    # Basic YOLO inference settings
    source: str = args.source
    weights: List[str] = args.weights
    view_img: bool = args.view_img
    imgsz: int = args.img_size
    results_loc: str = args.results_loc
    save_vid_or_img: bool = args.save

    # Check if source is webcam/stream
    webcam: bool = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://"))
    )

    # Create save directory
    save_dir: Path = Path(
        increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging and device
    set_logging()
    device: torch.device = select_device(args.device)
    half: bool = device.type != "cpu"  # half precision only supported on CUDA

    # Load the YOLO model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride: int = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # convert to FP16

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # speed up if image size remains constant
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get class names
    names = model.module.names if hasattr(model, "module") else model.names
    random_colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Warmup if using CUDA
    if device.type != "cpu":
        _ = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0: float = time.time()
    vid_path: Optional[str] = None
    vid_writer: Optional[cv2.VideoWriter] = None

    # Counters
    in_count: int = 0
    out_count: int = 0
    prev_path: Optional[str] = None

    for path, img, im0s, vid_cap in dataset:
        # Convert numpy (HWC) to torch tensor (CHW) and normalize
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        start_time: float = time_synchronized()
        pred = model(img_tensor, augment=args.augment)[0]

        # Apply non-maximum suppression
        detections_per_image = non_max_suppression(
            prediction=pred,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            classes=args.classes,
            agnostic=args.agnostic_nms
        )
        end_time: float = time_synchronized()

        # Prepare arrays for Deep SORT
        bboxes: List[List[float]] = []
        scores: List[float] = []
        class_names: List[str] = []

        # Process each detection
        for i, det in enumerate(detections_per_image):
            if webcam:
                # Multiple camera streams
                p, _, im0, frame_idx = path[i], f"{i}: ", im0s[i].copy(), dataset.count
            else:
                p, _, im0, frame_idx = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)

            if len(det):
                # Rescale coordinates from tensor image size to original image size
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls_idx in reversed(det):
                    # Convert bbox from xyxy to xywh
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()

                    # Track only specific class if needed. Here we assume all valid (or e.g., heads, persons)
                    # If you want to track only "head" (class idx = 1), you can check:
                    # if cls_idx.item() == 1: ...
                    bboxes.append(xywh)
                    scores.append(conf.item())
                    class_names.append(names[int(cls_idx.item())])

                # Convert (cx, cy, w, h) -> (x, y, w, h) for Deep SORT
                for box in bboxes:
                    box[0] -= box[2] / 2
                    box[1] -= box[3] / 2

        # Perform Deep SORT tracking
        np_bboxes = np.array(bboxes)
        np_scores = np.array(scores)

        features = encoder(im0, np_bboxes)
        detections_list = [
            Detection(bbox, score, cls_name, feature)
            for bbox, score, cls_name, feature in zip(np_bboxes, np_scores, class_names, features)
        ]

        # Non-max suppression for tracker-level
        boxs = np.array([d.tlwh for d in detections_list])
        track_scores = np.array([d.confidence for d in detections_list])
        track_classes = np.array([d.class_name for d in detections_list])
        indices = preprocessing.non_max_suppression(
            boxs, track_classes, nms_max_overlap, track_scores
        )
        detections_list = [detections_list[i] for i in indices]

        # Prepare boundary line (horizontal line at y = height//2)
        width: int = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if vid_cap else im0.shape[1]
        height: int = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if vid_cap else im0.shape[0]
        line_y: int = height // 2

        tracker.predict()
        tracker.update(detections_list, line_y_coord=line_y)

        # Visualization and counting
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox_tlbr = track.to_tlbr()  # (x1, y1, x2, y2)
            class_name = track.get_class()

            center_x = int((bbox_tlbr[0] + bbox_tlbr[2]) / 2)
            center_y = int((bbox_tlbr[1] + bbox_tlbr[3]) / 2)
            dist_from_line = center_y - line_y
            is_below_line = dist_from_line > 0

            # Count crossing
            if not track.below_line and is_below_line:
                if not track.stop_tracking:
                    in_count += 1
                    track.stop_tracking = True
                    track.below_line = is_below_line

            elif track.below_line and not is_below_line:
                if not track.stop_tracking:
                    out_count += 1
                    track.stop_tracking = True
                    track.below_line = is_below_line

            # Update track's below_line status
            track.below_line = is_below_line

            # Determine color: green if "stopped tracking & below line",
            # blue if "stopped tracking & above line", else black
            color: Tuple[int, int, int]
            if track.stop_tracking:
                color = (0, 255, 0) if track.below_line else (255, 0, 0)
            else:
                color = (0, 0, 0)

            label = f"{class_name}: {track.track_id}"
            plot_one_box(
                x=bbox_tlbr,
                img=im0,
                color=color,
                label=label,
                line_thickness=2,
                show_center=False
            )

            # Draw center as a small white dot
            cv2.circle(
                im0,
                center=(center_x, center_y),
                radius=3,
                color=(255, 255, 255),
                thickness=-1
            )

        # Draw the dividing line
        cv2.line(
            img=im0,
            pt1=(0, line_y),
            pt2=(width, line_y),
            color=(0, 155, 255),
            thickness=2,
        )

        # Display in/out counts
        cv2.putText(
            img=im0,
            text=f"in: {in_count}, out: {out_count}",
            org=(15, 195),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 255, 255),
            thickness=2
        )

        # If view_img is set, show results
        if view_img:
            screen_width = 1920  # Adjust as needed
            screen_height = 1080  # Adjust as needed
            scale_factor = min(screen_width / im0.shape[1], screen_height / im0.shape[0])
            resized_w = int(im0.shape[1] * scale_factor)
            resized_h = int(im0.shape[0] * scale_factor)
            resized_frame = cv2.resize(im0, (resized_w, resized_h))

            cv2.imshow(str(p), resized_frame)
            cv2.waitKey(1)

        # Save results (image or video)
        if save_vid_or_img:
            if dataset.mode == "image":
                cv2.imwrite(save_path, im0)
            else:
                # For video
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()

                    fourcc = "mp4v"  # output codec
                    fps: float = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30.0
                    w, h = width, height
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                    )
                if vid_writer:
                    vid_writer.write(im0)

        # Once we move to a new file (e.g. new video), write the counts from the old one
        if path != prev_path:
            if prev_path is not None:
                vid_name = prev_path.split("/")[-1]
                print(f"{vid_name} done")
                with open(f"{results_loc}/results.txt", "a") as f:
                    f.write(f"{vid_name} {in_count} {out_count}\n")

                # Reset for the next file
                in_count = 0
                out_count = 0

            prev_path = path

        # Print FPS
        fps: float = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0.0
        print(f"FPS: {fps:.2f}")

    # After finishing all frames
    if prev_path is not None:
        vid_name = prev_path.split("/")[-1]
        print(f"{vid_name} done")
        # Write final counts
        with open(f"{results_loc}/results.txt", "a") as f:
            f.write(f"{vid_name} {in_count} {out_count}\n")

    print(f"Done. ({time.time() - t0:.3f}s)")


def main() -> None:
    args = parse_args()
    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            # If you have multiple weights, you could loop here
            detect(args)
            for w in args.weights:
                strip_optimizer(w)
        else:
            detect(args)


if __name__ == "__main__":
    main()
