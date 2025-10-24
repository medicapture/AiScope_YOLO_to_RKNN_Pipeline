import argparse
import logging
import os
import platform

import colorlog
import torch
import torch.nn as nn
import matplotlib

matplotlib.use(
    "Agg"
)  # Use non-interactive backend for matplotlib, to prevent RuntimeError: main thread is not in main loop

from ultralytics import SETTINGS, YOLO, __version__

# Get current working directory
current_dir = os.getcwd()
# Update Ultralytics settings
SETTINGS.update({"datasets_dir": current_dir})

from ultralytics.nn.modules import Bottleneck, Conv

from ultralytics.utils import (
    DEFAULT_CFG,
    __version__,
)

# Set up logging configuration
def setup_logger(log_file=None, log_level=logging.INFO):
    """Configure logging with optional file output and specified log level."""
    logger = logging.getLogger("model_converter")
    logger.setLevel(log_level)

    # Create colored formatter for console
    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    # Create standard formatter for file (no colors in file)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified (no colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

# Initialize logger
logger = setup_logger(log_file="model_conversion.log")

def is_wsl():
    """Detect if running under Windows Subsystem for Linux (WSL)"""
    # Method 1: Check /proc/version for Microsoft signature
    try:
        with open("/proc/version", "r") as f:
            version_info = f.read().lower()
            if "microsoft" in version_info:
                return True
    except:
        pass

    # Method 2: Check kernel release string
    try:
        kernel = platform.release().lower()
        if "microsoft" in kernel or "-microsoft-" in kernel:
            return True
    except:
        pass

    # Method 3: Check for WSLInterop file
    if os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"):
        return True

    # Method 4: Check for WSL environment variable
    if os.environ.get("WSL_DISTRO_NAME"):
        return True

    return False

# Check if running on WSL2 and set environment variable
if is_wsl():
    logger.info("WSL2 detected. Disable pin_memory for PyTorch.")
    os.environ["PIN_MEMORY"] = (
        "false"  # Disable pin_memory for PyTorch for running on wsl2, since it is not supported
    )

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1, act=nn.ReLU())
        self.cv1 = Conv(c1, self.c, 1, 1, act=nn.ReLU())
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=nn.ReLU()) 
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def replace_activations(model, act = nn.LeakyReLU()):
    """Recursively convert all activation functions to specified activation in a YOLO model"""

    def replace_activations_recursive(module):
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.SiLU, nn.Hardswish, nn.LeakyReLU, nn.ReLU6, nn.SiLU)):
                # Replace SiLU/Swish with specified activation
                logger.info(f"Converted {name}: {type(child).__name__} → {type(act).__name__}")
                setattr(module, name, act)
            elif hasattr(child, 'act') and isinstance(child.act, (nn.ReLU, nn.SiLU, nn.Hardswish, nn.LeakyReLU, nn.ReLU6, nn.SiLU)):
                # Handle Conv wrapper modules that have 'act' attribute
                logger.info(f"Converted {name}.act: {type(child.act).__name__} → {type(act).__name__}")
                child.act = act
            else:
                # Recursively process child modules
                replace_activations_recursive(child)

    logger.info(f"Converting all activations to {type(act).__name__}...")
    replace_activations_recursive(model)
    logger.info(f"All activations converted to {type(act).__name__}")

def detect_pruned(model):
    """Detect if a YOLO model has been pruned by checking for C2f_v2 modules."""
    for name, module in model.model.named_modules():
        if isinstance(module, C2f_v2):
            logger.info(f"Pruned module detected: {name} is C2f_v2")
            return True
    logger.info("No pruned modules detected.")
    return False

def fine_tune(base_model):
    """
    Fine-tune a YOLO model on a custom dataset.
    The fine-tuned model will then go through conversion to RKNN format.
    """
    model = YOLO(
        base_model
    )  
    # Replace all activation functions with LeakyReLU for better RKNN compatibility
    replace_activations(model)
    
    if "data" in model.overrides:
        del model.overrides["data"]
    if "imgsz" in model.overrides:
        del model.overrides["imgsz"]
    if "single_cls" in model.overrides:
        del model.overrides["single_cls"]
    if detect_pruned(model):
        results = model.train(pruning=True)
    else:
        results = model.train()

    return results.save_dir

def export(model, img_size=None):
    """Export a YOLO model to a specific format."""
    cfg = DEFAULT_CFG
    cfg.model = model
    cfg.format = cfg.format or "torchscript"
    if img_size is not None:
        cfg.imgsz = img_size
    logger.debug(f"cfg = {cfg}")
    from ultralytics.models.yolo.model import YOLO

    # from ultralytics import YOLO
    model = YOLO(cfg.model)
    task = model.task
    model.export(**vars(cfg))  # Export to ONNX format with specified image size
    return task

def remove_softmax(model):
    # Remove Softmax layer from ONNX model for cls
    import onnx
    import onnx_graphsurgeon as gs

    INPUT_MODEL_PATH = model
    if not INPUT_MODEL_PATH.endswith(".onnx"):
        raise ValueError("Input model must be an ONNX file.")
    # Path to save the modified model
    OUTPUT_MODEL_PATH = (
        INPUT_MODEL_PATH[:-5] + "_softmax_removed" + INPUT_MODEL_PATH[-5:]
    )
    TARGET_NODE_NAME = "/model.10/Softmax"

    # --- 1. Load the Model and Isolate the Graph ---
    logger.info(f"Loading ONNX model from {INPUT_MODEL_PATH}")
    graph = gs.import_onnx(onnx.load(INPUT_MODEL_PATH))

    softmax_node = None
    for node in graph.nodes:
        # The 'op' attribute of the node corresponds to the layer type
        if node.name == TARGET_NODE_NAME:
            # We assume there is only one Softmax node at the end of the model.
            # If there are multiple, you might need more specific logic here.
            softmax_node = node
            logger.info(f"Found Softmax node: {softmax_node.name}")
            break

    if softmax_node is None:
        logger.error("Error: Could not find a Softmax node in the graph.")
        raise ValueError("Softmax node not found in the graph. Please check the model.")
    else:
        # --- 3. Rewire the Graph ---
        # The goal is to make the input of the Softmax node the new graph output.

        # Get the tensor that is the *input* to the Softmax node.
        # This is the output from the previous layer (the 'Gemm' layer in your case).
        previous_tensor = softmax_node.inputs[0]

        previous_tensor.dtype = "float32"
        previous_tensor.shape = [1, 1000]
        logger.info(
            f"Copied dtype ({previous_tensor.dtype}) and shape ({previous_tensor.shape}) to the new output tensor."
        )

        # Set this tensor as the new output of the entire graph.
        graph.outputs = [previous_tensor]
        logger.info(
            f"Set graph output to '{previous_tensor.name}', the input of the old Softmax node."
        )

        # --- 4. Clean Up the Graph ---
        # The cleanup() function will automatically remove any nodes that are no longer
        # connected to the graph's outputs. This includes our dangling Softmax node.
        graph.cleanup()
        logger.info("Graph cleaned up. The dangling Softmax node has been removed.")

        # --- 5. Export the Modified Graph to a New ONNX File ---
        onnx.save(gs.export_onnx(graph), OUTPUT_MODEL_PATH)
        logger.info(
            f"Success! Model without Softmax layer saved to: {OUTPUT_MODEL_PATH}"
        )

    # --- Verification Step (Optional but Recommended) ---
    logger.info("\nTo verify the change, load the new model with a tool like Netron.")
    logger.info(
        f"The model's output should now be the tensor that previously fed into the Softmax layer."
    )
    return OUTPUT_MODEL_PATH

def convert(model_path, platform, do_quant, output_path):
    from rknn.api import RKNN
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    logger.info("--> Config model")
    rknn.config(
        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], 
        target_platform=platform,
    )
    logger.info("Configuration complete")

    # Load model
    logger.info("--> Loading model")
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        logger.error(f"Load model failed with error code {ret}!")
        raise RuntimeError(f"Load model failed with error code {ret}!")
    logger.info("Model loading complete")

    # Build model
    logger.info("--> Building model")

    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        logger.error(f"Build model failed with error code {ret}!")
        raise RuntimeError(f"Build model failed with error code {ret}!")
    logger.info("Model building complete")

    # Export rknn model
    logger.info("--> Export rknn model")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        logger.error(f"Export rknn model failed with error code {ret}!")
        raise RuntimeError(f"Export rknn model failed with error code {ret}!")
    logger.info(f"Model successfully exported to {output_path}")

    # Release
    rknn.release()

# Define arguments
parser = argparse.ArgumentParser()

# arguments for model path
parser.add_argument(
    "--model", type=str, default="pretrained_models/yolov8s_unpruned.pt", help="Model path (original)"
)

# arguments for fine-tuning
parser.add_argument("--fine-tune", action="store_true", help="Enable fine-tuning")

# arguments for conversion
parser.add_argument(
    "--img-size",
    nargs='+',
    type=int,
    default=None,
    help="Image size for ONNX export, e.g., --img-size 640 or --img-size 384 640, this will override the default image size in default.yaml for ONNX export"
)
parser.add_argument(
    "--quant-dataset",
    type=str,
    default="./quantization_dataset/dataset.txt",
    help="Path to quantization dataset",
)
parser.add_argument(
    "--platform",
    type=str,
    default="rk3588",
    help="Target platform",
    choices=["rk3562", "rk3566", "rk3568", "rk3576", "rk3588"],
)

if __name__ == "__main__":
    """
    Usage example for basic conversion: `python aiscope_pipeline.py --model pretrained_weights/yolo11s_unpruned.pt`
    Usage example for fine-tuning + conversion: `python aiscope_pipeline.py --model pretrained_weights/yolo11s_unpruned.pt --fine-tune`
    Remember to prepare finetuning dataset in YOLO format before running fine-tuning.
    Modify the default configs in 'ultralytics/cfg/default.yaml' if necessary.
    """
    # Parse arguments
    args = parser.parse_args()
    if args.img_size:
        if len(args.img_size) > 2:
            raise ValueError("img-size must be a single integer or two integers (height width).")
    if args.fine_tune:
        # Fine-tune the model
        logger.info("=" * 20 + "Fine-tuning is enabled, start fine-tuning" + "=" * 20)
        logger.info(f"Base model path: {args.model}")
        args.model = str(fine_tune(args.model) / "weights/best.pt")
        logger.info(f"Fine-tuned model path: {args.model}")

    # Conversion
    # To ONNX
    logger.info("=" * 20 + "Exporting to ONNX" + "=" * 20)
    task = export(args.model, args.img_size)
    logger.info(
        f"ONNX export complete. Model saved to {args.model.replace('.pt', '.onnx')}"
    )
    if task == "classify":
        logger.info("=" * 20 + "Removing Softmax layer" + "=" * 20)
        args.model = remove_softmax(args.model.replace(".pt", ".onnx"))
        # args.model = args.model[:-3] + "_softmax_removed" + args.model[-3:]

    # To RKNN
    logger.info("=" * 20 + "Converting to RKNN" + "=" * 20)

    # do_quant = args.model_type == "i8"
    DATASET_PATH = args.quant_dataset
    logger.info(f"Quantization dataset: {DATASET_PATH}")
    
    output_path = args.model.split(".")[0] + "_" + args.model_type + ".rknn"
    model = args.model.replace(".pt", ".onnx")
    convert(model, args.platform, True, output_path)

    logger.info("=" * 20 + "Conversion finished successfully" + "=" * 20)
    logger.info(f"Converted model is saved to {output_path}.")
    logger.info("To test the model, please deploy it on RKNN platform.")