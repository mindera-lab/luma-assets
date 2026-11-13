#!/usr/bin/env python3
"""Reproducible Supertonic 3 weight compression. No credentials or user data.

CPU computation stays FP32. Large constant weights are stored in FP16 and cast
back by ONNX Runtime; optional dynamic INT8 MatMul quantization is evaluated
separately. Every artifact retains the upstream model license and modification
notice. Run with Python 3.12, onnx 1.19.1, onnxruntime 1.23.2, numpy 2.3.4.
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import shutil
import urllib.request

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

REVISION = 'aafc6e32416a594460b32413efc49d7fe4ce6d46'
SOURCE = 'https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/' + REVISION + '/'
FILES = {
    'onnx/duration_predictor.onnx': 'c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db',
    'onnx/text_encoder.onnx': 'c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff',
    'onnx/vector_estimator.onnx': '883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c',
    'onnx/vocoder.onnx': '085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba',
    'onnx/tts.json': '42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09',
    'onnx/unicode_indexer.json': '9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f',
    'voice_styles/F1.json': 'bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2',
    'LICENSE': '0d944a9110fed9a9602d60e0423a272903e7bd21ab060490774efc77c2275e9f',
}

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def package_file(destination, path):
    packed = destination.with_name(destination.name + '.gz')
    with destination.open('rb') as incoming, packed.open('wb') as output:
        with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0, compresslevel=9) as compressor:
            shutil.copyfileobj(incoming, compressor)
    return {'path': path, 'asset': path.replace('/', '--') + '.gz', 'bytes': destination.stat().st_size,
            'sha256': digest(destination), 'downloadBytes': packed.stat().st_size,
            'downloadSHA256': digest(packed), 'compression': 'gzip'}

def compress_model(source, destination, int8):
    if int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(source), str(destination), weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul'], per_channel=True)
        model = onnx.load(destination)
    else:
        model = onnx.load(source)
    casts, weights, changed = [], [], 0
    for tensor in model.graph.initializer:
        data = numpy_helper.to_array(tensor)
        # Small constants include quantization scales, timing and normalization.
        # Keeping them exact avoids overflow/underflow and preserves graph control.
        if tensor.data_type == TensorProto.FLOAT and data.size >= 1024 and np.isfinite(data).all() and np.max(np.abs(data)) < 65504:
            name = tensor.name + '__luma_fp16'
            weights.append(numpy_helper.from_array(data.astype(np.float16), name))
            casts.append(helper.make_node('Cast', [name], [tensor.name], to=TensorProto.FLOAT, name=name + '_restore'))
            changed += 1
        else:
            weights.append(tensor)
    del model.graph.initializer[:]
    model.graph.initializer.extend(weights)
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(casts + nodes)
    note = 'Modified by LUMA: FP16 weight storage with FP32 computation' + (' and per-channel INT8 MatMul weights.' if int8 else '.')
    model.doc_string = note + ' Source revision ' + REVISION + '. BigScience Open RAIL-M terms, including Attachment A, apply.'
    model.metadata_props.add(key='luma.modification', value=note)
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    ort.InferenceSession(str(destination), sess_options=options, providers=['CPUExecutionProvider'])
    return changed

def build(args):
    out, source = pathlib.Path(args.output), pathlib.Path(args.source)
    out.mkdir(parents=True, exist_ok=True); source.mkdir(parents=True, exist_ok=True)
    entries = []
    for path, expected in FILES.items():
        original = source / pathlib.Path(path).name
        if not original.exists():
            urllib.request.urlretrieve(SOURCE + path, original)
        if digest(original) != expected:
            raise ValueError('Upstream checksum mismatch: ' + path)
        destination = out / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith('.onnx'):
            changed = compress_model(original, destination, args.int8)
            print(path, changed, destination.stat().st_size, flush=True)
        else:
            shutil.copyfile(original, destination)
        entries.append(package_file(destination, path))
    notice = ('LUMA compact Supertonic 3\n\nModified weight files: FP16 storage restored to FP32 for CPU inference'
              + ('; per-channel dynamic INT8 MatMul quantization' if args.int8 else '')
              + '.\nOriginal source: ' + SOURCE + '\n\nAll model files, modified or unmodified, remain subject to the attached BigScience Open RAIL-M LICENSE, including paragraph 5 and Attachment A use restrictions. This package is not covered by the exercise repository\'s Unlicense. No endorsement by Supertone is claimed.\n')
    (out / 'NOTICE.txt').write_text(notice)
    entries.append(package_file(out / 'NOTICE.txt', 'NOTICE.txt'))
    manifest = {'version': 1, 'id': 'supertonic-3-' + ('int8-fp16' if args.int8 else 'fp16') + '-v1', 'upstreamRevision': REVISION,
                'license': 'BigScience-Open-RAIL-M', 'files': entries, 'downloadBytes': sum(e['downloadBytes'] for e in entries)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'id': manifest['id'], 'downloadBytes': manifest['downloadBytes']}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='.build/supertonic')
    parser.add_argument('--output', required=True)
    parser.add_argument('--int8', action='store_true')
    build(parser.parse_args())
