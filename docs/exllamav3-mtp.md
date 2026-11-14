# Integrated MTP in ExLlamaV3

The native ExLlamaV3 loader offers `--exllamav3-mtp` (or the `exllamav3-mtp` checkbox). It is off by default. Use `--draft-max` to set the number of proposed tokens, at least 1. Do not select an external `model-draft` at the same time. This integration targets the Qwen head stored under `mtp.*`; checking `mtp.fc.weight` or `mtp.fc.trellis` establishes that the input projection exists, not that every required tensor is present or valid. A partial checkpoint will fail during loading. The option requires an ExLlamaV3 version exposing the MTP component and interfaces described below; it was implemented against v1.6.6, not verified on other releases.

## Engine findings

In [v1.6.6](https://github.com/turboderp-org/exllamav3/tree/bd1f519172406a9560bd5c90e61000691bb69a07), [Qwen configuration](https://github.com/turboderp-org/exllamav3/blob/v1.6.6/exllamav3/architecture/qwen3_5.py) registers the `mtp` component based on config, not the existence of the weights. The [Qwen MTP component](https://github.com/turboderp-org/exllamav3/blob/v1.6.6/exllamav3/architecture/qwen3_5_mtp.py) attaches to the target and borrows its embedding and output head. The [reference loader](https://github.com/turboderp-org/exllamav3/blob/v1.6.6/exllamav3/model_init.py) creates a separate draft model and cache, loads the draft first, then passes it to target loading. [Generator](https://github.com/turboderp-org/exllamav3/blob/v1.6.6/exllamav3/generator/generator.py) attaches the component and needs enough recurrent cache history for draft verification. The cache must be created before loading its model. This integration follows that sequence and releases models on failure or unload.

The Qwen MTP component declares `supports_tp: False`: it is loaded in layer-split mode, while the target can use TP. The target's `lm_head_argmax` implements a TP path. The source therefore provides the required wiring for a TP target with this non-TP draft head; that is not a hardware validation. It consumes additional memory for the head, its cache, and recurrent history. No engine changes are required by the inspected interface.

## Hardware validation

1. With ExLlamaV3 1.6.6, load the same Qwen3.8-27B EXL3 checkpoint using the native ExLlamaV3 loader, Q8 cache, desired context and native TP. Leave `model-draft` empty and `exllamav3-mtp` unchecked. Generate a fixed prompt and record output, throughput, VRAM and any errors.
2. Unload, enable `exllamav3-mtp`, set `draft-max` to 3, reload, and generate the same prompt with the same sampling parameters. Check the `Loading integrated MTP component` log, successful loading, coherent output and VRAM usage. A matching output is only expected for deterministic settings, and can still differ due to numerical effects.
3. Repeat with short and long contexts and multiple requests, then unload and reload. Compare throughput only after warmup. If VRAM is insufficient, reduce context or draft depth and retry. Record errors and configuration; no speedup is guaranteed.

These checks have not been run on the requested GPU pair or checkpoint. A reported upstream [TP + MTP concurrency issue](https://github.com/turboderp-org/exllamav3/issues/474) concerns v1.5.3 and does not establish its status in v1.6.6.
