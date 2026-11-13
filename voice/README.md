# LUMA compact Supertonic 3

This package stores large Supertonic 3 weights in FP16, restores them to FP32 for CPU computation, and uses gzip for transport. The initial local build transfers about 185.5 MB instead of 398.8 MB and installs about 200.6 MB. This is not an INT8 model and does not promise a corresponding reduction in runtime memory.

Source: https://huggingface.co/supertone-oss-archive/supertonic-3/tree/aafc6e32416a594460b32413efc49d7fe4ce6d46

All original and modified model files are distributed under **BigScience Open RAIL-M**, including paragraph 5 and Attachment A use restrictions. These restrictions continue to apply to downstream use and redistribution. Read the attached LICENSE and NOTICE.txt before using or redistributing the model. The exercise dataset's Unlicense does not apply to these files. No endorsement by Supertone is claimed.

Modified by LUMA: FP16 weight storage with explicit FP32 restoration nodes. Each modified ONNX file contains a modification notice. Tokenization, voice style and configuration are unchanged. The application downloads the listed gzip files when a supported language plan is added and validates both compressed and expanded sizes and SHA-256 digests before activating the package.

The build uses checksum-pinned source weights and pinned Python dependencies. `manifest.json` identifies every installed and transferred file. CI checks the resulting model hashes against the reviewed manifest, opens all four graphs with ONNX Runtime CPU, and verifies every gzip roundtrip before publishing. Published v1 assets are never overwritten; future changes require a new release identifier.

Local regression: deterministic five-step English, Spanish, Japanese and Korean samples preserved output duration; waveform correlations against the FP32 baseline were 0.986, 0.992, 0.9995 and 0.9993 respectively. This is a small numerical regression set, not a listening study. Mandarin uses the platform voice because this Supertonic model does not support Chinese.
