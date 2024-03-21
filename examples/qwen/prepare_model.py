import argparse
import os

from tensorrt_llm.logger import logger

def add_model_argument(parser: argparse.ArgumentParser):
    """Add parameters for download upload model from www.modelscope.cn

    Args:
        parser (argparse.ArgumentParser): The command argument parser.
    """
    parser.add_argument('--model', 
                        type=str, 
                        required=True,
                        help='Specify model id of www.modelscope.cn, or model directory')
    parser.add_argument('--revision', 
                        type=str, 
                        default=None,
                        help='Specify the model version')   
    
 
def prepare_model_files(model_id: str,
                       revision: str = None,
                       cache_dir: str = None)->str:
    """Download model from www.modelscope.cn

    Args:
        model_id (str): The model if
        revision (str, optional): The model revision. Defaults to None.
        cache_dir (str, optional): The model cache dir. Defaults to None.

    Returns:
        str: The downloaded model path.
    """
    try:
        from modelscope import snapshot_download
        if not os.path.exists(model_id):
            model_path = snapshot_download(model_id=model_id,
                                           cache_dir=cache_dir,
                                           revision=revision)
            return model_path
    except ImportError as e:
        logger.warn(
            "Use model from www.modelscope.cn need pip install modelscope"
        )
        raise e
    
    