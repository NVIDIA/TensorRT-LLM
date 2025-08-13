from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class QuantAnnotation:
    """Annotation for quantization metadata.
    
    Attributes:
        quant_scheme: Quantization scheme (e.g., 'int8', 'fp8')
        backend: Quantization backend implementation
        quant_group: Group ID for shard-aware quantization
        metadata: Additional quantization parameters
    """
    quant_scheme: str
    backend: str
    quant_group: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

def annotate_node(node, annotation: QuantAnnotation):
    """Annotate a node with quantization metadata"""
    if not hasattr(node, 'meta'):
        node.meta = {}
    node.meta['quant_annotation'] = annotation

def get_quant_annotation(node) -> Optional[QuantAnnotation]:
    """Retrieve quantization annotation from a node"""
    return node.meta.get('quant_annotation', None)

def has_quant_annotation(node) -> bool:
    """Check if a node has quantization annotation"""
    return get_quant_annotation(node) is not None

def clear_quant_annotation(node):
    """Remove quantization annotation from a node"""
    if hasattr(node, 'meta') and 'quant_annotation' in node.meta:
        del node.meta['quant_annotation']

def copy_quant_annotation(source_node, target_node):
    """Copy quantization annotation from source node to target node"""
    annotation = get_quant_annotation(source_node)
    if annotation:
        annotate_node(target_node, annotation)

def update_annotation_for_sharding(node, shard_info: Dict[str, Any]):
    """Update quantization annotation to include sharding information"""
    annotation = get_quant_annotation(node)
    if annotation:
        # Update metadata with sharding info
        if annotation.metadata is None:
            annotation.metadata = {}
        annotation.metadata.update(shard_info)
        # Update quant_group if sharding affects quantization grouping
        if 'shard_rank' in shard_info:
            annotation.quant_group = shard_info['shard_rank']